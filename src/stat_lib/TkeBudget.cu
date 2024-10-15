#include "TkeBudget.cuh"
#include <filesystem>
#include <mpi.h>
#include "../gxl_lib/MyString.h"

MPI_Offset cfd::create_tke_budget_file(cfd::Parameter &parameter, const Mesh &mesh, int n_block_ahead) {
  const std::filesystem::path out_dir("output/stat");
  MPI_File fp;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/tke_budget.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
  MPI_Status status;
  MPI_Offset offset{0};

  const int myid = parameter.get_int("myid");
  constexpr int n_collect = TkeBudget::n_collect;
  constexpr int ngg = TkeBudget::ngg;
  if (myid == 0) {
    const int n_block = mesh.n_block_total;
    // collect reynolds 1st order statistics
    MPI_File_write_at(fp, 0, &n_block, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp, 4, &n_collect, 1, MPI_INT32_T, &status);
    MPI_File_write_at(fp, 8, &ngg, 1, MPI_INT32_T, &status);
    offset = 4 * 3;
    for (auto var: TkeBudget::collect_name) {
      gxl::write_str(var.data(), fp, offset);
    }
  }
  MPI_Bcast(&offset, 1, MPI_OFFSET, 0, MPI_COMM_WORLD);

  if (myid != 0) {
    offset += 4 * n_collect; // counter
  }
  for (int b = 0; b < n_block_ahead; ++b) {
    MPI_Offset sz = (mesh.mx_blk[b] + 2 * ngg) * (mesh.my_blk[b] + 2 * ngg) * (mesh.mz_blk[b] + 2 * ngg) * 8;
    offset += sz * n_collect + 4 * 3;
  }

  return offset;
}

std::vector<int> cfd::read_tke_budget_file(cfd::Parameter &parameter, const cfd::Mesh &mesh, int n_block_ahead,
                                           std::vector<Field> &field) {
  const std::filesystem::path out_dir("output/stat");
  std::string file_name = (out_dir.string() + "/tke_budget.bin");

  std::vector<int> counter_read(TkeBudget::n_collect, 0);
  if (!std::filesystem::exists(file_name)) {
    printf("File %s does not exist. The Tke budgets are collected from now\n", file_name.c_str());
  } else {
    MPI_File fp_rey1;
    MPI_File_open(MPI_COMM_WORLD, file_name.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fp_rey1);
    MPI_Status status;

    constexpr int n_collect = TkeBudget::n_collect;
    constexpr int ngg = TkeBudget::ngg;
    const auto &reyAveVar = TkeBudget::collect_name;

    MPI_Offset offset_read;
    int nBlock = 0;
    MPI_File_read_at(fp_rey1, 0, &nBlock, 1, MPI_INT32_T, &status);
    if (nBlock != mesh.n_block_total) {
      printf("Error: The number of blocks in %s is not consistent with the current mesh.\n", file_name.c_str());
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int n_read_rey1 = 0;
    MPI_File_read_at(fp_rey1, 4, &n_read_rey1, 1, MPI_INT32_T, &status);
    int ngg_read = 0;
    MPI_File_read_at(fp_rey1, 8, &ngg_read, 1, MPI_INT32_T, &status);
    if (ngg_read != TkeBudget::ngg) {
      printf("Error: The number of ghost cells in %s is not %d.\n", file_name.c_str(), TkeBudget::ngg);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<std::string> rey1Var(n_read_rey1);
    offset_read = 4 * 3;
    for (int l = 0; l < n_read_rey1; ++l) {
      rey1Var[l] = gxl::read_str_from_binary_MPI_ver(fp_rey1, offset_read);
    }
    std::vector<int> counter_read_temp(n_read_rey1);
    MPI_File_read_at(fp_rey1, offset_read, counter_read_temp.data(), n_read_rey1, MPI_INT32_T, &status);
    offset_read += 4 * n_read_rey1;
    std::vector<int> read_rey1_index(n_read_rey1, -1);
    for (int l = 0; l < n_read_rey1; ++l) {
      for (int i = 0; i < n_collect; ++i) {
        if (rey1Var[l] == reyAveVar[i]) {
          read_rey1_index[l] = i;
          counter_read[i] = counter_read_temp[l];
          break;
        }
      }
    }

    for (int b = 0; b < n_block_ahead; ++b) {
      MPI_Offset sz = (mesh.mx_blk[b] + 2 * ngg) * (mesh.my_blk[b] + 2 * ngg) * (mesh.mz_blk[b] + 2 * ngg) * 8;
      offset_read += sz * n_read_rey1 + 4 * 3;
    }

    for (int b = 0; b < mesh.n_block; ++b) {
      int mx, my, mz;
      MPI_File_read_at(fp_rey1, offset_read, &mx, 1, MPI_INT32_T, &status);
      offset_read += 4;
      MPI_File_read_at(fp_rey1, offset_read, &my, 1, MPI_INT32_T, &status);
      offset_read += 4;
      MPI_File_read_at(fp_rey1, offset_read, &mz, 1, MPI_INT32_T, &status);
      offset_read += 4;
      if (mx != mesh[b].mx || my != mesh[b].my || mz != mesh[b].mz) {
        printf("Error: The mesh size in %s is not consistent with the current mesh.\n", file_name.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      const auto sz = (long long) (mx + 2 * ngg) * (my + 2 * ngg) * (mz + 2 * ngg) * 8;
      MPI_Datatype ty;
      int lSize[3]{mx + 2 * ngg, my + 2 * ngg, mz + 2 * ngg};
      int start_idx[3]{0, 0, 0};
      MPI_Type_create_subarray(3, lSize, lSize, start_idx, MPI_ORDER_FORTRAN, MPI_DOUBLE, &ty);
      MPI_Type_commit(&ty);

      MPI_File_read_at(fp_rey1, offset_read, field[b].collect_tke_budget.data(), n_collect, ty, &status);
      cudaMemcpy(field[b].h_ptr->collect_tke_budget.data(), field[b].collect_tke_budget.data(), sz * n_collect,
                 cudaMemcpyHostToDevice);
    }
  }

  return counter_read;
}

__device__ void cfd::collect_tke_budget(cfd::DZone *zone, cfd::DParameter *param, int i, int j, int k) {

}

void
cfd::export_tke_budget_file(Parameter &parameter, const Mesh &mesh, std::vector<Field> &field, MPI_Offset offset_ahead,
                            std::vector<int> &counter, const std::vector<MPI_Datatype> &tys) {
  const std::filesystem::path out_dir("output/stat");
  MPI_File fp_rey1;
  MPI_File_open(MPI_COMM_WORLD, (out_dir.string() + "/tke_budget.bin").c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fp_rey1);
  MPI_Status status;

  MPI_Offset offset = offset_ahead;
  constexpr int n_collect = TkeBudget::n_collect;
  const int myid = parameter.get_int("myid");
  if (myid == 0) {
    MPI_File_write_at(fp_rey1, offset, counter.data(), n_collect, MPI_INT32_T, &status);
    offset += 4 * n_collect;
  }
  constexpr int ngg = TkeBudget::ngg;
  for (int b = 0; b < mesh.n_block; ++b) {
    auto &zone = field[b].h_ptr;
    const int mx = mesh[b].mx, my = mesh[b].my, mz = mesh[b].mz;
    const auto sz = (long long) (mx + 2 * ngg) * (my + 2 * ngg) * (mz + 2 * ngg) * 8;

    cudaMemcpy(field[b].collect_tke_budget.data(), zone->collect_tke_budget.data(), sz * n_collect,
               cudaMemcpyDeviceToHost);

    // We create this datatype because in the original MPI_File_write_at, the number of elements is a 64-bit integer.
    // However, the nx*ny*nz may be larger than 2^31, so we need to use MPI_Type_create_subarray to create a datatype
    MPI_Datatype ty = tys[b];

    MPI_File_write_at(fp_rey1, offset, &mx, 1, MPI_INT32_T, &status);
    offset += 4;
    MPI_File_write_at(fp_rey1, offset, &my, 1, MPI_INT32_T, &status);
    offset += 4;
    MPI_File_write_at(fp_rey1, offset, &mz, 1, MPI_INT32_T, &status);
    offset += 4;
    MPI_File_write_at(fp_rey1, offset, field[b].collect_tke_budget.data(), n_collect, ty, &status);
    offset += sz * n_collect;
  }
}