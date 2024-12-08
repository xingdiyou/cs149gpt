#include <ATen/ATen.h>
#include <immintrin.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>

// Uncomment for ISPC
// #include "module_ispc.h"
// using namespace ispc;

// ------------------------------------ //
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, int x, int y,
                        const int sizeX) {
  // Note that sizeX is the size of a Row, not the number of rows
  return tensor[x * (sizeX) + y];
}

inline void twoDimWrite(std::vector<float> &tensor, int x, int y,
                        const int sizeX, float val) {
  tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, int x, int y, int z, int b,
                         const int &sizeX, const int &sizeY, const int &sizeZ) {
  return tensor[x * sizeX * sizeY * sizeZ + y * sizeY * sizeZ + z * sizeZ + b];
}

inline void fourDimWrite(std::vector<float> &tensor, int x, int y, int z, int b,
                         const int sizeX, const int sizeY, const int sizeZ,
                         float val) {
  tensor[x * sizeX * sizeY * sizeZ + y * sizeY * sizeZ + z * sizeZ + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
  tensor = tensor.flatten();
  tensor = tensor.contiguous();
  std::vector<float> vec(tensor.data_ptr<float>(),
                         tensor.data_ptr<float>() + tensor.numel());
  return vec;
}

/* Programming Your Attention Modules.
 *
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We
 * have also created O and QK^t Tensors that are formatted as vectors. After you
 * have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: batchSize, numHeads, seqLen,
 * dim:
 *
 * batchSize - The number of samples for your attention layer. Think of it this
 * way - if I asked my dnn a question and it output 5 different answers it had a
 * batch size of 5. These samples are independent of each other and thus can be
 * parallelized.
 *
 * numHeads - Each head runs on its own set of Q, K, V matrices. This
 * effectively allows each head to operate the same attention algorithm, but
 * each with each head using different hyperparameters. These allow each head to
 * have their own definition of what relevance is when looking at a token. These
 * heads can operate independently of one another and thus can be parallized.
 *
 * seqLen - The number of tokens. You may think of this as the number of words
 * in a sample.
 *
 * dim - The number of features each token encodes per attention head. Let's
 * say I encoded a word using the follow (length, number of vowels, has a
 * capital letters). The emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor,
                               torch::Tensor VTensor, torch::Tensor QK_tTensor,
                               int batchSize, int numHeads, int seqLen,
                               int dim) {
  // Q, K, V are passed in with Shape: (batchSize, numHeads, seqLen, dim)
  // QK^t Intermediate Tensor has Shape (seqLen, seqLen)

  // Make O Tensor with Shape (batchSize, numHeads, seqLen, dim)
  at::Tensor OTensor =
      at::zeros({batchSize, numHeads, seqLen, dim}, at::kFloat);

  // Format O, Q, K, and V tensors into 4D vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);

  // Format QK_t Tensor into a 2D vector.
  std::vector<float> QK_t = formatTensor(QK_tTensor);

  // -------- YOUR CODE HERE  -------- //

  for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    for (int headIdx = 0; headIdx < numHeads; headIdx++) {
      for (int queryIdx = 0; queryIdx < seqLen; queryIdx++) {
        for (int keyIdx = 0; keyIdx < seqLen; keyIdx++) {
          float score = 0.0f;
          for (int d = 0; d < dim; d++) {
            score += fourDimRead(Q, batchIdx, headIdx, queryIdx, d, numHeads,
                                 seqLen, dim) *
                     fourDimRead(K, batchIdx, headIdx, keyIdx, d, numHeads,
                                 seqLen, dim);
          }
          twoDimWrite(QK_t, queryIdx, keyIdx, seqLen, score);
        }
      }

      for (int queryIdx = 0; queryIdx < seqLen; queryIdx++) {
        float rowSum = 0.0f;
        for (int keyIdx = 0; keyIdx < seqLen; keyIdx++) {
          float score = std::exp(twoDimRead(QK_t, queryIdx, keyIdx, seqLen));
          twoDimWrite(QK_t, queryIdx, keyIdx, seqLen, score);
          rowSum += score;
        }
        for (int keyIdx = 0; keyIdx < seqLen; keyIdx++) {
          float prob = twoDimRead(QK_t, queryIdx, keyIdx, seqLen) / rowSum;
          twoDimWrite(QK_t, queryIdx, keyIdx, seqLen, prob);
        }
      }

      for (int queryIdx = 0; queryIdx < seqLen; queryIdx++) {
        for (int d = 0; d < dim; d++) {
          float output = 0.0f;
          for (int keyIdx = 0; keyIdx < seqLen; keyIdx++) {
            output += twoDimRead(QK_t, queryIdx, keyIdx, seqLen) *
                      fourDimRead(V, batchIdx, headIdx, keyIdx, d, numHeads,
                                  seqLen, dim);
          }
          fourDimWrite(O, batchIdx, headIdx, queryIdx, d, numHeads, seqLen, dim,
                       output);
        }
      }
    }
  }

  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (batchSize,
  // numHeads, seqLen, dim) and returns it //
  return torch::from_blob(O.data(), {batchSize, numHeads, seqLen, dim},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor,
                                        torch::Tensor KTensor,
                                        torch::Tensor VTensor,
                                        torch::Tensor QK_tTensor, int batchSize,
                                        int numHeads, int seqLen, int dim) {
  // Q, K, V are passed in with Shape: (batchSize, numHeads, seqLen, dim)
  // QK^t Intermediate Tensor has Shape (seqLen, seqLen)

  // Make O Tensor with Shape (batchSize, numHeads, seqLen, dim)
  at::Tensor OTensor =
      at::zeros({batchSize, numHeads, seqLen, dim}, at::kFloat);

  // Format O, Q, K, and V tensors into 4D vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);

  // Format QK_t Tensor into a 2D vector.
  std::vector<float> QK_t = formatTensor(QK_tTensor);

  // -------- YOUR CODE HERE  -------- //

  constexpr int TILE_SIZE = 16;

  for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    for (int headIdx = 0; headIdx < numHeads; headIdx++) {
      for (int queryStart = 0; queryStart < seqLen; queryStart += TILE_SIZE) {
        for (int keyStart = 0; keyStart < seqLen; keyStart += TILE_SIZE) {
          for (int dimStart = 0; dimStart < dim; dimStart += TILE_SIZE) {
            for (int queryIdx = queryStart;
                 queryIdx < std::min(queryStart + TILE_SIZE, seqLen);
                 queryIdx++) {
              for (int keyIdx = keyStart;
                   keyIdx < std::min(keyStart + TILE_SIZE, seqLen); keyIdx++) {
                float score = twoDimRead(QK_t, queryIdx, keyIdx, seqLen);
                for (int d = dimStart; d < std::min(dimStart + TILE_SIZE, dim);
                     d++) {
                  score += fourDimRead(Q, batchIdx, headIdx, queryIdx, d,
                                       numHeads, seqLen, dim) *
                           fourDimRead(K, batchIdx, headIdx, keyIdx, d,
                                       numHeads, seqLen, dim);
                }
                twoDimWrite(QK_t, queryIdx, keyIdx, seqLen, score);
              }
            }
          }
        }
      }

      for (int queryIdx = 0; queryIdx < seqLen; queryIdx++) {
        float rowSum = 0.0f;
        for (int keyIdx = 0; keyIdx < seqLen; keyIdx++) {
          float score = std::exp(twoDimRead(QK_t, queryIdx, keyIdx, seqLen));
          twoDimWrite(QK_t, queryIdx, keyIdx, seqLen, score);
          rowSum += score;
        }
        for (int keyIdx = 0; keyIdx < seqLen; keyIdx++) {
          float prob = twoDimRead(QK_t, queryIdx, keyIdx, seqLen) / rowSum;
          twoDimWrite(QK_t, queryIdx, keyIdx, seqLen, prob);
        }
      }

      for (int queryStart = 0; queryStart < seqLen; queryStart += TILE_SIZE) {
        for (int dimStart = 0; dimStart < dim; dimStart += TILE_SIZE) {
          for (int keyStart = 0; keyStart < seqLen; keyStart += TILE_SIZE) {
            for (int queryIdx = queryStart;
                 queryIdx < std::min(queryStart + TILE_SIZE, seqLen);
                 queryIdx++) {
              for (int d = dimStart; d < std::min(dimStart + TILE_SIZE, dim);
                   d++) {
                float output = fourDimRead(O, batchIdx, headIdx, queryIdx, d,
                                           numHeads, seqLen, dim);
                for (int keyIdx = keyStart;
                     keyIdx < std::min(keyStart + TILE_SIZE, seqLen);
                     keyIdx++) {
                  output += twoDimRead(QK_t, queryIdx, keyIdx, seqLen) *
                            fourDimRead(V, batchIdx, headIdx, keyIdx, d,
                                        numHeads, seqLen, dim);
                }
                fourDimWrite(O, batchIdx, headIdx, queryIdx, d, numHeads,
                             seqLen, dim, output);
              }
            }
          }
        }
      }
    }
  }

  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (batchSize,
  // numHeads, seqLen, d) and returns it //
  return torch::from_blob(O.data(), {batchSize, numHeads, seqLen, dim},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION     	              //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor,
                               torch::Tensor VTensor, torch::Tensor temp,
                               int batchSize, int numHeads, int seqLen,
                               int dim) {
  // Q, K, V are passed in with Shape: (batchSize, numHeads, seqLen, dim)

  // Make O Tensor with Shape (batchSize, numHeads, seqLen, dim)
  // and O Row Tensor with Shape (seqLen)
  at::Tensor OTensor =
      at::zeros({batchSize, numHeads, seqLen, dim}, at::kFloat);
  at::Tensor ORowTensor = at::zeros({seqLen}, at::kFloat);

  // Format Y, Q, K, and V tensors into 4D vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);

  // Format ORow Tensor into a 1D vector
  //  You can simply access this as ORow[i]
  std::vector<float> ORow = formatTensor(ORowTensor);

  // -------- YOUR CODE HERE  -------- //
  // We give you a template of the first three loops for your convenience
  // loop over batch
#pragma omp parallel for collapse(3)
  for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    // Loop over attention heads
    for (int headIdx = 0; headIdx < numHeads; headIdx++) {
      for (int queryIdx = 0; queryIdx < seqLen; queryIdx++) {
        // Local row buffer for attention scores, scoped per thread for OpenMP
        at::Tensor localRowTensor = temp.index({torch::indexing::Slice(
            omp_get_thread_num(), torch::indexing::None)});
        std::vector<float> ORow = formatTensor(localRowTensor);

        for (int keyIdx = 0; keyIdx < seqLen; keyIdx++) {
          float score = 0.0f;
          for (int d = 0; d < dim; d++) {
            score += fourDimRead(Q, batchIdx, headIdx, queryIdx, d, numHeads,
                                 seqLen, dim) *
                     fourDimRead(K, batchIdx, headIdx, keyIdx, d, numHeads,
                                 seqLen, dim);
          }
          ORow[keyIdx] = score;
        }

        float rowSum = 0.0f;

        for (int keyIdx = 0; keyIdx < seqLen; keyIdx++) {
          float score = std::exp(ORow[keyIdx]);
          ORow[keyIdx] = score;
          rowSum += score;
        }

        for (int d = 0; d < dim; d++) {
          float output = 0.0f;
          for (int keyIdx = 0; keyIdx < seqLen; keyIdx++) {
            output += ORow[keyIdx] / rowSum *
                      fourDimRead(V, batchIdx, headIdx, keyIdx, d, numHeads,
                                  seqLen, dim);
          }
          fourDimWrite(O, batchIdx, headIdx, queryIdx, d, numHeads, seqLen, dim,
                       output);
        }
      }
    }
  }

  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (batchSize,
  // numHeads, seqLen, d) and returns it //
  return torch::from_blob(O.data(), {batchSize, numHeads, seqLen, dim},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(
    torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor,
    torch::Tensor QiTensor, torch::Tensor KjTensor, torch::Tensor VjTensor,
    torch::Tensor SijTensor, torch::Tensor PijTensor, torch::Tensor PVTensor,
    torch::Tensor OiTensor, torch::Tensor LTensor, torch::Tensor LiTensor,
    torch::Tensor LijTensor, torch::Tensor LnewTensor, int Bc, int Br,
    int batchSize, int numHeads, int seqLen, int dim) {
  // Q, K, V are passed in with Shape: (batchSize, numHeads, seqLen, dim)
  // Sij, Pij are passed in with Shape: (Br, Bc)
  // Kj, Vj are passed in with Shape: (Bc, dim)
  // Qi, Oi, and PV  are passed in with Shape: (Br, dim)
  // L in passed in with Shape: (seqLen)
  // Li, Lij, and Lnew are passed in with shape (Br)

  // Make O Tensor with Shape (batchSize, numHeads, seqLen, dim)
  at::Tensor OTensor =
      at::zeros({batchSize, numHeads, seqLen, dim}, at::kFloat);

  // Format All Tensors into Vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);
  std::vector<float> Sij = formatTensor(SijTensor);
  std::vector<float> Pij = formatTensor(PijTensor);
  std::vector<float> Kj = formatTensor(KjTensor);
  std::vector<float> Vj = formatTensor(VjTensor);
  std::vector<float> Qi = formatTensor(QiTensor);
  std::vector<float> Oi = formatTensor(OiTensor);
  std::vector<float> l = formatTensor(LTensor);
  std::vector<float> PV = formatTensor(PVTensor);
  std::vector<float> li = formatTensor(LiTensor);
  std::vector<float> lij = formatTensor(LijTensor);
  std::vector<float> lnew = formatTensor(LnewTensor);

  // -------- YOUR CODE HERE  -------- //

  for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    for (int headIdx = 0; headIdx < numHeads; headIdx++) {
      std::fill(l.begin(), l.end(), 0);
      for (int keyStart = 0; keyStart < seqLen; keyStart += Bc) {
        for (int keyIdx = keyStart; keyIdx < std::min(keyStart + Bc, seqLen);
             keyIdx++) {
          for (int d = 0; d < dim; d++) {
            float key = fourDimRead(K, batchIdx, headIdx, keyIdx, d, numHeads,
                                    seqLen, dim);
            twoDimWrite(Kj, keyIdx - keyStart, d, dim, key);
            float val = fourDimRead(V, batchIdx, headIdx, keyIdx, d, numHeads,
                                    seqLen, dim);
            twoDimWrite(Vj, keyIdx - keyStart, d, dim, val);
          }
        }

        for (int queryStart = 0; queryStart < seqLen; queryStart += Br) {
          for (int queryIdx = queryStart;
               queryIdx < std::min(queryStart + Br, seqLen); queryIdx++) {
            for (int d = 0; d < dim; d++) {
              float query = fourDimRead(Q, batchIdx, headIdx, queryIdx, d,
                                        numHeads, seqLen, dim);
              twoDimWrite(Qi, queryIdx - queryStart, d, dim, query);
              float output = fourDimRead(O, batchIdx, headIdx, queryIdx, d,
                                         numHeads, seqLen, dim);
              twoDimWrite(Oi, queryIdx - queryStart, d, dim, output);
            }
            li[queryIdx - queryStart] = l[queryIdx];
          }

          for (int queryIdx = queryStart;
               queryIdx < std::min(queryStart + Br, seqLen); queryIdx++) {
            for (int keyIdx = keyStart;
                 keyIdx < std::min(keyStart + Bc, seqLen); keyIdx++) {
              float score = 0.0f;
              for (int d = 0; d < dim; d++) {
                float query = twoDimRead(Qi, queryIdx - queryStart, d, dim);
                float key = twoDimRead(Kj, keyIdx - keyStart, d, dim);
                score += query * key;
              }
              twoDimWrite(Sij, queryIdx - queryStart, keyIdx - keyStart, Bc,
                          score);
            }
          }

          for (int queryIdx = queryStart;
               queryIdx < std::min(queryStart + Br, seqLen); queryIdx++) {
            float rowSum = 0.0f;
            for (int keyIdx = keyStart;
                 keyIdx < std::min(keyStart + Bc, seqLen); keyIdx++) {
              float prob = std::exp(twoDimRead(Sij, queryIdx - queryStart,
                                               keyIdx - keyStart, Bc));
              twoDimWrite(Pij, queryIdx - queryStart, keyIdx - keyStart, Bc,
                          prob);
              rowSum += prob;
            }
            lij[queryIdx - queryStart] = rowSum;
            lnew[queryIdx - queryStart] =
                li[queryIdx - queryStart] + lij[queryIdx - queryStart];
          }

          for (int queryIdx = queryStart;
               queryIdx < std::min(queryStart + Br, seqLen); queryIdx++) {
            for (int d = 0; d < dim; d++) {
              float output = li[queryIdx - queryStart] *
                             twoDimRead(Oi, queryIdx - queryStart, d, dim);
              for (int keyIdx = keyStart;
                   keyIdx < std::min(keyStart + Bc, seqLen); keyIdx++) {
                float prob = twoDimRead(Pij, queryIdx - queryStart,
                                        keyIdx - keyStart, Bc);
                float val = twoDimRead(Vj, keyIdx - keyStart, d, dim);
                output += prob * val;
              }
              output /= lnew[queryIdx - queryStart];
              twoDimWrite(Oi, queryIdx - queryStart, d, dim, output);
            }
          }

          for (int queryIdx = queryStart;
               queryIdx < std::min(queryStart + Br, seqLen); queryIdx++) {
            for (int d = 0; d < dim; d++) {
              float output = twoDimRead(Oi, queryIdx - queryStart, d, dim);
              fourDimWrite(O, batchIdx, headIdx, queryIdx, d, numHeads, seqLen,
                           dim, output);
            }
            l[queryIdx] = lnew[queryIdx - queryStart];
          }
        }
      }
    }
  }

  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (batchSize,
  // numHeads, seqLen, d) and returns it //
  return torch::from_blob(O.data(), {batchSize, numHeads, seqLen, dim},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked,
        " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
