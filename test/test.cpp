#include "../src/types.h"
#include <iostream>
#include <sstream>
#include <chrono>


template <typename T>
void CHECK(const T &t, std::string s, std::string test_title = "")
{
    std::ostringstream oss;
    oss << t;
    std::cout << test_title << (oss.str() == s ? " PASS" : " FAIL") << std::endl;
}

int main() 
{
    std::vector<Float32> data(3 * 4 * 5, 0);
    for (int i = 0; i < 3 * 4 * 5; ++i)
        data[i] = static_cast<Float32>(i);

    Tensor<Float32> tensor_1({3, 4, 5}, data.data());
    CHECK(tensor_1,
R"(Tensor([[[0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00],
[5.0000e+00, 6.0000e+00, 7.0000e+00, 8.0000e+00, 9.0000e+00],
[1.0000e+01, 1.1000e+01, 1.2000e+01, 1.3000e+01, 1.4000e+01],
[1.5000e+01, 1.6000e+01, 1.7000e+01, 1.8000e+01, 1.9000e+01]],

[[2.0000e+01, 2.1000e+01, 2.2000e+01, 2.3000e+01, 2.4000e+01],
[2.5000e+01, 2.6000e+01, 2.7000e+01, 2.8000e+01, 2.9000e+01],
[3.0000e+01, 3.1000e+01, 3.2000e+01, 3.3000e+01, 3.4000e+01],
[3.5000e+01, 3.6000e+01, 3.7000e+01, 3.8000e+01, 3.9000e+01]],

[[4.0000e+01, 4.1000e+01, 4.2000e+01, 4.3000e+01, 4.4000e+01],
[4.5000e+01, 4.6000e+01, 4.7000e+01, 4.8000e+01, 4.9000e+01],
[5.0000e+01, 5.1000e+01, 5.2000e+01, 5.3000e+01, 5.4000e+01],
[5.5000e+01, 5.6000e+01, 5.7000e+01, 5.8000e+01, 5.9000e+01]]], dtype=DType, dims=[3, 4, 5])
)"
    , "tensor_1");


    CHECK(tensor_1.transpose(1, 2),
R"(Tensor([[[0.0000e+00, 5.0000e+00, 1.0000e+01, 1.5000e+01],
[1.0000e+00, 6.0000e+00, 1.1000e+01, 1.6000e+01],
[2.0000e+00, 7.0000e+00, 1.2000e+01, 1.7000e+01],
[3.0000e+00, 8.0000e+00, 1.3000e+01, 1.8000e+01],
[4.0000e+00, 9.0000e+00, 1.4000e+01, 1.9000e+01]],

[[2.0000e+01, 2.5000e+01, 3.0000e+01, 3.5000e+01],
[2.1000e+01, 2.6000e+01, 3.1000e+01, 3.6000e+01],
[2.2000e+01, 2.7000e+01, 3.2000e+01, 3.7000e+01],
[2.3000e+01, 2.8000e+01, 3.3000e+01, 3.8000e+01],
[2.4000e+01, 2.9000e+01, 3.4000e+01, 3.9000e+01]],

[[4.0000e+01, 4.5000e+01, 5.0000e+01, 5.5000e+01],
[4.1000e+01, 4.6000e+01, 5.1000e+01, 5.6000e+01],
[4.2000e+01, 4.7000e+01, 5.2000e+01, 5.7000e+01],
[4.3000e+01, 4.8000e+01, 5.3000e+01, 5.8000e+01],
[4.4000e+01, 4.9000e+01, 5.4000e+01, 5.9000e+01]]], dtype=DType, dims=[3, 5, 4])
)"
    , "tensor_1.transpose(1, 2)");


    Tensor<Float32> tensor_2({3, 5, 4}, data.data());
    CHECK(MatMul(tensor_1, tensor_2), 
R"(Tensor([[[1.2000e+02, 1.3000e+02, 1.4000e+02, 1.5000e+02],
[3.2000e+02, 3.5500e+02, 3.9000e+02, 4.2500e+02],
[5.2000e+02, 5.8000e+02, 6.4000e+02, 7.0000e+02],
[7.2000e+02, 8.0500e+02, 8.9000e+02, 9.7500e+02]],

[[3.1200e+03, 3.2300e+03, 3.3400e+03, 3.4500e+03],
[3.8200e+03, 3.9550e+03, 4.0900e+03, 4.2250e+03],
[4.5200e+03, 4.6800e+03, 4.8400e+03, 5.0000e+03],
[5.2200e+03, 5.4050e+03, 5.5900e+03, 5.7750e+03]],

[[1.0120e+04, 1.0330e+04, 1.0540e+04, 1.0750e+04],
[1.1320e+04, 1.1555e+04, 1.1790e+04, 1.2025e+04],
[1.2520e+04, 1.2780e+04, 1.3040e+04, 1.3300e+04],
[1.3720e+04, 1.4005e+04, 1.4290e+04, 1.4575e+04]]], dtype=DType, dims=[3, 4, 4])
)"
    , "MatMul(tensor_1, tensor_2)");


    TensorMap tensor_2_map = tensor_2.map(0);
    CHECK(MatMul(tensor_1, tensor_2_map),
R"(Tensor([[[1.2000e+02, 1.3000e+02, 1.4000e+02, 1.5000e+02],
[3.2000e+02, 3.5500e+02, 3.9000e+02, 4.2500e+02],
[5.2000e+02, 5.8000e+02, 6.4000e+02, 7.0000e+02],
[7.2000e+02, 8.0500e+02, 8.9000e+02, 9.7500e+02]],

[[9.2000e+02, 1.0300e+03, 1.1400e+03, 1.2500e+03],
[1.1200e+03, 1.2550e+03, 1.3900e+03, 1.5250e+03],
[1.3200e+03, 1.4800e+03, 1.6400e+03, 1.8000e+03],
[1.5200e+03, 1.7050e+03, 1.8900e+03, 2.0750e+03]],

[[1.7200e+03, 1.9300e+03, 2.1400e+03, 2.3500e+03],
[1.9200e+03, 2.1550e+03, 2.3900e+03, 2.6250e+03],
[2.1200e+03, 2.3800e+03, 2.6400e+03, 2.9000e+03],
[2.3200e+03, 2.6050e+03, 2.8900e+03, 3.1750e+03]]], dtype=DType, dims=[3, 4, 4])
)"
    , "MatMul(tensor_1, tensor_2_map)");


    Tensor<Float32> tensor_3({3, 1, 5}, data.data());
    CHECK(tensor_1 + tensor_3,
R"(Tensor([[[0.0000e+00, 2.0000e+00, 4.0000e+00, 6.0000e+00, 8.0000e+00],
[5.0000e+00, 7.0000e+00, 9.0000e+00, 1.1000e+01, 1.3000e+01],
[1.0000e+01, 1.2000e+01, 1.4000e+01, 1.6000e+01, 1.8000e+01],
[1.5000e+01, 1.7000e+01, 1.9000e+01, 2.1000e+01, 2.3000e+01]],

[[2.5000e+01, 2.7000e+01, 2.9000e+01, 3.1000e+01, 3.3000e+01],
[3.0000e+01, 3.2000e+01, 3.4000e+01, 3.6000e+01, 3.8000e+01],
[3.5000e+01, 3.7000e+01, 3.9000e+01, 4.1000e+01, 4.3000e+01],
[4.0000e+01, 4.2000e+01, 4.4000e+01, 4.6000e+01, 4.8000e+01]],

[[5.0000e+01, 5.2000e+01, 5.4000e+01, 5.6000e+01, 5.8000e+01],
[5.5000e+01, 5.7000e+01, 5.9000e+01, 6.1000e+01, 6.3000e+01],
[6.0000e+01, 6.2000e+01, 6.4000e+01, 6.6000e+01, 6.8000e+01],
[6.5000e+01, 6.7000e+01, 6.9000e+01, 7.1000e+01, 7.3000e+01]]], dtype=DType, dims=[3, 4, 5])
)"
    , "tensor_1 + tensor_3");


    Tensor<Float32> tensor_4({3, 4, 5}, data.data());
    CHECK(tensor_4.softmax(),
R"(Tensor([[[1.1656e-02, 3.1685e-02, 8.6129e-02, 2.3412e-01, 6.3641e-01],
[1.1656e-02, 3.1685e-02, 8.6129e-02, 2.3412e-01, 6.3641e-01],
[1.1656e-02, 3.1685e-02, 8.6129e-02, 2.3412e-01, 6.3641e-01],
[1.1656e-02, 3.1685e-02, 8.6129e-02, 2.3412e-01, 6.3641e-01]],

[[1.1656e-02, 3.1685e-02, 8.6129e-02, 2.3412e-01, 6.3641e-01],
[1.1656e-02, 3.1685e-02, 8.6129e-02, 2.3412e-01, 6.3641e-01],
[1.1656e-02, 3.1685e-02, 8.6129e-02, 2.3412e-01, 6.3641e-01],
[1.1656e-02, 3.1685e-02, 8.6129e-02, 2.3412e-01, 6.3641e-01]],

[[1.1656e-02, 3.1685e-02, 8.6129e-02, 2.3412e-01, 6.3641e-01],
[1.1656e-02, 3.1685e-02, 8.6129e-02, 2.3412e-01, 6.3641e-01],
[1.1656e-02, 3.1685e-02, 8.6129e-02, 2.3412e-01, 6.3641e-01],
[1.1656e-02, 3.1685e-02, 8.6129e-02, 2.3412e-01, 6.3641e-01]]], dtype=DType, dims=[3, 4, 5])
)"
    , "tensor_4.softmax()");


    Linear<Float32> linear(2, 3, true);
    for (int i = 0; i < 3 * 2; ++i)
        linear.weight.data()[i] = static_cast<Float32>(i + 1);
    for (int i = 0; i < 3; ++i)
        linear.bias.data()[i] = static_cast<Float32>(i + 1);
    Tensor<Float32> tensor_5 = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6};
    tensor_5.view({2, 3, 2});
    CHECK(linear(tensor_5), 
R"(Tensor([[[4.0000e+00, 9.0000e+00, 1.4000e+01],
[7.0000e+00, 1.6000e+01, 2.5000e+01],
[1.0000e+01, 2.3000e+01, 3.6000e+01]],

[[1.3000e+01, 3.0000e+01, 4.7000e+01],
[1.6000e+01, 3.7000e+01, 5.8000e+01],
[1.9000e+01, 4.4000e+01, 6.9000e+01]]], dtype=DType, dims=[2, 3, 3])
)"
    , "linear(tensor_5)");


    Tensor<Float32> tensor_6(3, 4, 5);
    for (int i = 0; i < 3 * 4 * 5; ++i)
        tensor_6.data()[i] = static_cast<Float32>(i) * (i % 2 == 0 ? -1 : 1);
    GELU<Float32> gelu;
    CHECK(gelu(tensor_6),
R"(Tensor([[[-0.0000e+00, 8.4119e-01, -4.5402e-02, 2.9964e+00, -7.0214e-05],
[5.0000e+00, -0.0000e+00, 7.0000e+00, -0.0000e+00, 9.0000e+00],
[-0.0000e+00, 1.1000e+01, -0.0000e+00, 1.3000e+01, -0.0000e+00],
[1.5000e+01, -0.0000e+00, 1.7000e+01, -0.0000e+00, 1.9000e+01]],

[[-0.0000e+00, 2.1000e+01, -0.0000e+00, 2.3000e+01, -0.0000e+00],
[2.5000e+01, -0.0000e+00, 2.7000e+01, -0.0000e+00, 2.9000e+01],
[-0.0000e+00, 3.1000e+01, -0.0000e+00, 3.3000e+01, -0.0000e+00],
[3.5000e+01, -0.0000e+00, 3.7000e+01, -0.0000e+00, 3.9000e+01]],

[[-0.0000e+00, 4.1000e+01, -0.0000e+00, 4.3000e+01, -0.0000e+00],
[4.5000e+01, -0.0000e+00, 4.7000e+01, -0.0000e+00, 4.9000e+01],
[-0.0000e+00, 5.1000e+01, -0.0000e+00, 5.3000e+01, -0.0000e+00],
[5.5000e+01, -0.0000e+00, 5.7000e+01, -0.0000e+00, 5.9000e+01]]], dtype=DType, dims=[3, 4, 5])
)"
    , "gelu(tensor_6)");
    CHECK(tensor_6.mean(true),
R"(Tensor([[[-4.0000e-01],
[1.4000e+00],
[-2.4000e+00],
[3.4000e+00]],

[[-4.4000e+00],
[5.4000e+00],
[-6.4000e+00],
[7.4000e+00]],

[[-8.4000e+00],
[9.4000e+00],
[-1.0400e+01],
[1.1400e+01]]], dtype=DType, dims=[3, 4, 1])
)"
    , "tensor_6.mean()");
    CHECK(tensor_6.var(true, false),
R"(Tensor([[[5.8400e+00],
[4.9040e+01],
[1.4024e+02],
[2.7944e+02]],

[[4.6664e+02],
[7.0184e+02],
[9.8504e+02],
[1.3162e+03]],

[[1.6954e+03],
[2.1226e+03],
[2.5978e+03],
[3.1210e+03]]], dtype=DType, dims=[3, 4, 1])
)"
    , "tensor_6.var(true, false)");
    CHECK(tensor_6.var(),
R"(Tensor([[7.3000e+00, 6.1300e+01, 1.7530e+02, 3.4930e+02],
[5.8330e+02, 8.7730e+02, 1.2313e+03, 1.6453e+03],
[2.1193e+03, 2.6533e+03, 3.2473e+03, 3.9013e+03]], dtype=DType, dims=[3, 4])
)"
    , "tensor_6.var()");


    auto params = load_parameters("gpt2-small-124M.bin", false);
    MultiHeadAttention<Float32> mha(768, 768, 1024, 12, true);
    mha.m_mask = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.mask"].tensor);
    mha.m_W_query.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_query.weight"].tensor);
    mha.m_W_query.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_query.bias"].tensor);
    mha.m_W_key.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_key.weight"].tensor);
    mha.m_W_key.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_key.bias"].tensor);
    mha.m_W_value.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_value.weight"].tensor);
    mha.m_W_value.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_value.bias"].tensor);
    mha.m_out_proj.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.out_proj.weight"].tensor);
    mha.m_out_proj.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.out_proj.bias"].tensor);

    Tensor<Float32> tensor_x(2, 4, 768);
    for (int i = 0; i < 2 * 4 * 768; ++i)
        tensor_x.data()[i] = static_cast<Float32>(i) / 1000.0f;
    CHECK(mha(tensor_x, false),
R"(Tensor([[[2.7979e+00, 8.6255e+00, -1.9196e+00, ..., 1.5051e+00, 3.6155e-01, 3.3295e-01],
[9.8455e+00, 1.7378e+01, -2.9882e+00, ..., 3.2588e+00, -4.8087e-01, 1.4326e+00],
[1.6892e+01, 2.6130e+01, -4.0577e+00, ..., 5.0113e+00, -1.3233e+00, 2.5320e+00],
[2.3940e+01, 3.4882e+01, -5.1267e+00, ..., 6.7643e+00, -2.1658e+00, 3.6315e+00]],

[[1.4782e+01, 4.6968e+01, -1.7866e+01, ..., 1.4725e+01, 7.6113e-01, 4.6302e+00],
[2.1829e+01, 5.5720e+01, -1.8935e+01, ..., 1.6478e+01, -8.1313e-02, 5.7297e+00],
[2.8877e+01, 6.4473e+01, -2.0004e+01, ..., 1.8231e+01, -9.2376e-01, 6.8293e+00],
[3.5924e+01, 7.3225e+01, -2.1073e+01, ..., 1.9984e+01, -1.7662e+00, 7.9288e+00]]], dtype=DType, dims=[2, 4, 768])
)"
    , "mha(tensor_x, false)");


    TransformerBlock<Float32> trf_block(768, 768, 1024, 12, true);
    trf_block.m_att.m_mask = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.mask"].tensor);
    trf_block.m_att.m_W_query.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_query.weight"].tensor);
    trf_block.m_att.m_W_query.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_query.bias"].tensor);
    trf_block.m_att.m_W_key.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_key.weight"].tensor);
    trf_block.m_att.m_W_key.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_key.bias"].tensor);
    trf_block.m_att.m_W_value.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_value.weight"].tensor);
    trf_block.m_att.m_W_value.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_value.bias"].tensor);
    trf_block.m_att.m_out_proj.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.out_proj.weight"].tensor);
    trf_block.m_att.m_out_proj.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.out_proj.bias"].tensor);
    trf_block.m_ff.m_linear1.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.ff.layers.0.weight"].tensor);
    trf_block.m_ff.m_linear1.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.ff.layers.0.bias"].tensor);
    trf_block.m_ff.m_linear2.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.ff.layers.2.weight"].tensor);
    trf_block.m_ff.m_linear2.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.ff.layers.2.bias"].tensor);
    trf_block.m_norm1.m_scale = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.norm1.scale"].tensor);
    trf_block.m_norm1.m_shift = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.norm1.shift"].tensor);
    trf_block.m_norm2.m_scale = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.norm2.scale"].tensor);
    trf_block.m_norm2.m_shift = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.norm2.shift"].tensor);
    CHECK(trf_block(tensor_x, false),
R"(Tensor([[[1.9627e+00, 1.6904e+00, 1.2804e+00, ..., -2.8107e+00, 1.8830e+00, 1.4044e+00],
[2.7307e+00, 2.4584e+00, 2.0484e+00, ..., -2.0427e+00, 2.6510e+00, 2.1724e+00],
[3.4987e+00, 3.2264e+00, 2.8164e+00, ..., -1.2747e+00, 3.4190e+00, 2.9404e+00],
[4.2667e+00, 3.9944e+00, 3.5844e+00, ..., -5.0670e-01, 4.1870e+00, 3.7084e+00]],

[[5.0348e+00, 4.7624e+00, 4.3524e+00, ..., 2.6129e-01, 4.9550e+00, 4.4764e+00],
[5.8028e+00, 5.5304e+00, 5.1204e+00, ..., 1.0293e+00, 5.7230e+00, 5.2444e+00],
[6.5707e+00, 6.2984e+00, 5.8884e+00, ..., 1.7973e+00, 6.4910e+00, 6.0124e+00],
[7.3387e+00, 7.0664e+00, 6.6564e+00, ..., 2.5653e+00, 7.2590e+00, 6.7804e+00]]], dtype=DType, dims=[2, 4, 768])
)"
    , "trf_block(tensor_x, false)");


    GPTModel<Float32> gpt(50257, 768, 1024, 12, 12, true);
    gpt.m_tok_embedding.weight = *static_cast<Tensor<Float32>*>(params["tok_emb.weight"].tensor);
    gpt.m_pos_embedding.weight = *static_cast<Tensor<Float32>*>(params["pos_emb.weight"].tensor);
    for (size_t i = 0; i < gpt.m_trf_blocks.size(); ++i)
    {
        auto& trf_block = gpt.m_trf_blocks[i];
        trf_block.m_att.m_mask = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".att.mask"].tensor);
        trf_block.m_att.m_W_query.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".att.W_query.weight"].tensor);
        trf_block.m_att.m_W_query.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".att.W_query.bias"].tensor);
        trf_block.m_att.m_W_key.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".att.W_key.weight"].tensor);
        trf_block.m_att.m_W_key.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".att.W_key.bias"].tensor);
        trf_block.m_att.m_W_value.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".att.W_value.weight"].tensor);
        trf_block.m_att.m_W_value.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".att.W_value.bias"].tensor);
        trf_block.m_att.m_out_proj.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".att.out_proj.weight"].tensor);
        trf_block.m_att.m_out_proj.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".att.out_proj.bias"].tensor);
        trf_block.m_ff.m_linear1.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".ff.layers.0.weight"].tensor);
        trf_block.m_ff.m_linear1.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".ff.layers.0.bias"].tensor);
        trf_block.m_ff.m_linear2.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".ff.layers.2.weight"].tensor);
        trf_block.m_ff.m_linear2.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".ff.layers.2.bias"].tensor);
        trf_block.m_norm1.m_scale = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".norm1.scale"].tensor);
        trf_block.m_norm1.m_shift = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".norm1.shift"].tensor);
        trf_block.m_norm2.m_scale = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".norm2.scale"].tensor);
        trf_block.m_norm2.m_shift = *static_cast<Tensor<Float32>*>(params["trf_blocks." + std::to_string(i) + ".norm2.shift"].tensor);
    }
    gpt.m_final_norm.m_scale = *static_cast<Tensor<Float32>*>(params["final_norm.scale"].tensor);
    gpt.m_final_norm.m_shift = *static_cast<Tensor<Float32>*>(params["final_norm.shift"].tensor);
    gpt.m_out_head.weight = *static_cast<Tensor<Float32>*>(params["out_head.weight"].tensor);

    Tensor<TokenIndex> token_ids = {{1, 2, 3, 4}};
    auto beg = std::chrono::steady_clock::now();
    auto output = gpt(token_ids);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - beg;
    CHECK(output,
R"(Tensor([[[-3.2901e+01, -3.1202e+01, -3.4662e+01, ..., -3.9487e+01, -3.9873e+01, -3.2239e+01],
[-5.5521e+01, -5.3429e+01, -5.6477e+01, ..., -6.8154e+01, -6.6771e+01, -5.8601e+01],
[-6.1797e+01, -6.0539e+01, -5.9550e+01, ..., -7.5321e+01, -7.2773e+01, -6.5571e+01],
[-6.6029e+01, -6.6317e+01, -6.2770e+01, ..., -7.9216e+01, -7.7247e+01, -6.8870e+01]]], dtype=DType, dims=[1, 4, 50257])
)"
    , "gpt(token_ids)");
    std::cout << "GPT forward elapsed = " << elapsed.count() << " ms\n";

    return 0;
}