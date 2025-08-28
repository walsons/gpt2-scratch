#include "../src/types.h"
#include <iostream>
#include <chrono>


int main() 
{
    // Float32* data = new Float32[3 * 4 * 5];
    // for (int i = 0; i < 3 * 4 * 5; ++i)
    //     data[i] = static_cast<Float32>(i);

    // Tensor tensor({3, 4, 5}, data, false);
    // std::cout << tensor << std::endl;

    // Tensor transposed_tensor = tensor;
    // transposed_tensor.transpose(1, 2);
    // std::cout << transposed_tensor << std::endl;

    // Tensor tensor2({3, 5, 4}, data, false);
    // std::cout << MatMul(tensor, tensor2) << std::endl;

    // Tensor tensor3 = tensor2[0];
    // std::cout << MatMul(tensor, tensor3) << std::endl;

    // Tensor tensor4({3, 1, 5}, data, false);
    // std::cout << (tensor + tensor4) << std::endl;

    // Linear<Float32> linear(2, 3, true);
    // Float32* linear_data = new Float32[3 * 2];
    // for (int i = 0; i < 3 * 2; ++i)
    //     linear_data[i] = static_cast<Float32>(i + 1);
    // linear.weight = Tensor<Float32>({3, 2}, linear_data, false);
    // linear.bias = Tensor<Float32>({3}, linear_data, false);
    // Float32 input_data[] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6};
    // Tensor input_x({2, 3, 2}, input_data);
    // std::cout << linear(input_x) << std::endl;

    // Tensor tensor5({3, 4, 5}, data);
    // tensor5.softmax();
    // std::cout << tensor5 << std::endl;


    // auto params = load_parameters("gpt2-small-124M.bin", true);
    // MultiHeadAttention<Float32> mha(768, 768, 1024, 12, true);
    // mha.m_mask = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.mask"].tensor);
    // mha.m_W_query.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_query.weight"].tensor);
    // mha.m_W_query.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_query.bias"].tensor);
    // mha.m_W_key.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_key.weight"].tensor);
    // mha.m_W_key.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_key.bias"].tensor);
    // mha.m_W_value.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_value.weight"].tensor);
    // mha.m_W_value.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_value.bias"].tensor);
    // mha.m_out_proj.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.out_proj.weight"].tensor);
    // mha.m_out_proj.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.out_proj.bias"].tensor);

    // Float32* input_data1 = new Float32[2 * 4 * 768];
    // for (int i = 0; i < 2 * 4 * 768; ++i)
    //     input_data1[i] = static_cast<Float32>(i) / 1000.0f;
    // Tensor<Float32> input_x1({2, 4, 768}, input_data1);
    // std::cout << mha(input_x1) << std::endl;
    /*
    tensor([[[  2.7979,   8.6255,  -1.9196,  ...,   1.5051,   0.3615,   0.3330],
         [  9.8455,  17.3776,  -2.9882,  ...,   3.2588,  -0.4809,   1.4326],
         [ 16.8926,  26.1301,  -4.0577,  ...,   5.0113,  -1.3233,   2.5320],
         [ 23.9399,  34.8825,  -5.1267,  ...,   6.7643,  -2.1658,   3.6315]],

        [[ 14.7821,  46.9678, -17.8664,  ...,  14.7249,   0.7611,   4.6302],
         [ 21.8295,  55.7201, -18.9354,  ...,  16.4780,  -0.0813,   5.7297],
         [ 28.8769,  64.4725, -20.0044,  ...,  18.2310,  -0.9238,   6.8293],
         [ 35.9242,  73.2249, -21.0735,  ...,  19.9841,  -1.7662,   7.9288]]],
       grad_fn=<ViewBackward0>)
    */

    // Float32* data = new Float32[3 * 4 * 5];
    // for (int i = 0; i < 3 * 4 * 5; ++i)
    //     data[i] = static_cast<Float32>(i) * (i % 2 == 0 ? -1 : 1);
    // Tensor<Float32> x({3, 4, 5}, data);
    // Tensor<Float32> x1({3, 4, 5}, data);
    // Tensor<Float32> x2({3, 4, 5}, data);
    // Tensor<Float32> x3({3, 4, 5}, data);
    // GELU<Float32> gelu;
    // std::cout << gelu(x) << std::endl;
    // std::cout << x1.mean(true) << std::endl;
    // std::cout << x2.var(true, false) << std::endl;
    // std::cout << x3.var() << std::endl;


    // auto params = load_parameters("gpt2-small-124M.bin", true);
    // TransformerBlock<Float32> trf_block(768, 768, 1024, 12, true);
    // trf_block.m_att.m_mask = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.mask"].tensor);
    // trf_block.m_att.m_W_query.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_query.weight"].tensor);
    // trf_block.m_att.m_W_query.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_query.bias"].tensor);
    // trf_block.m_att.m_W_key.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_key.weight"].tensor);
    // trf_block.m_att.m_W_key.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_key.bias"].tensor);
    // trf_block.m_att.m_W_value.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_value.weight"].tensor);
    // trf_block.m_att.m_W_value.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.W_value.bias"].tensor);
    // trf_block.m_att.m_out_proj.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.out_proj.weight"].tensor);
    // trf_block.m_att.m_out_proj.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.att.out_proj.bias"].tensor);
    // trf_block.m_ff.m_linear1.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.ff.layers.0.weight"].tensor);
    // trf_block.m_ff.m_linear1.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.ff.layers.0.bias"].tensor);
    // trf_block.m_ff.m_linear2.weight = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.ff.layers.2.weight"].tensor);
    // trf_block.m_ff.m_linear2.bias = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.ff.layers.2.bias"].tensor);
    // trf_block.m_norm1.m_scale = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.norm1.scale"].tensor);
    // trf_block.m_norm1.m_shift = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.norm1.shift"].tensor);
    // trf_block.m_norm2.m_scale = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.norm2.scale"].tensor);
    // trf_block.m_norm2.m_shift = *static_cast<Tensor<Float32>*>(params["trf_blocks.0.norm2.shift"].tensor);

    // Float32* input_data1 = new Float32[2 * 4 * 768];
    // for (int i = 0; i < 2 * 4 * 768; ++i)
    //     input_data1[i] = static_cast<Float32>(i) / 1000.0f;
    // Tensor<Float32> input_x1({2, 4, 768}, input_data1);
    // std::cout << trf_block(input_x1) << std::endl;
    /*
    tensor([[[ 1.9627,  1.6904,  1.2804,  ..., -2.8107,  1.8830,  1.4044],
         [ 2.7307,  2.4584,  2.0484,  ..., -2.0427,  2.6510,  2.1724],
         [ 3.4988,  3.2264,  2.8164,  ..., -1.2747,  3.4190,  2.9404],
         [ 4.2667,  3.9944,  3.5844,  ..., -0.5067,  4.1870,  3.7084]],

        [[ 5.0347,  4.7624,  4.3524,  ...,  0.2613,  4.9550,  4.4764],
         [ 5.8028,  5.5304,  5.1204,  ...,  1.0293,  5.7230,  5.2444],
         [ 6.5707,  6.2984,  5.8884,  ...,  1.7973,  6.4910,  6.0124],
         [ 7.3387,  7.0664,  6.6564,  ...,  2.5653,  7.2590,  6.7804]]],
       grad_fn=<AddBackward0>)
    */


    auto params = load_parameters("gpt2-small-124M.bin", true);
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

    TokenIndex input_data[] = {1, 2, 3, 4};
    auto t0 = std::chrono::steady_clock::now();
    auto output = gpt(Tensor<TokenIndex>({1, 4}, input_data));
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> ms = t1 - t0;
    std::cout << output << std::endl;
    std::cout << "elapsed = " << ms.count() << " ms\n";
    /*
    tensor([[[-32.9010, -31.2024, -34.6622,  ..., -39.4867, -39.8731, -32.2387],
         [-55.5208, -53.4285, -56.4767,  ..., -68.1539, -66.7709, -58.6006],
         [-61.7968, -60.5386, -59.5503,  ..., -75.3206, -72.7731, -65.5706],
         [-66.0290, -66.3174, -62.7706,  ..., -79.2162, -77.2467, -68.8703]]],
       grad_fn=<UnsafeViewBackward0>)
    */

    return 0;
}