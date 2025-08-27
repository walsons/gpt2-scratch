#include "types.h"
#include "bpe_tokenizer.h"
#include <iostream>


int main() 
{
    auto params = load_parameters("gpt2-small-124M.bin");

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

    BPETokenizer tokenizer("vocab.json", "merges.txt", {"<|endoftext|>"});

    std::string text = "there is album";
    auto token_ids = tokenizer.encode(text);

    auto next_token_id = [&](std::vector<TokenIndex> token_ids) -> TokenIndex {
        auto output = gpt(Tensor<TokenIndex>({1, static_cast<typename Tensor<TokenIndex>::Dim>(token_ids.size())}, token_ids.data()));
        auto &final_output = output[0][output.dims()[1] - 1];
        size_t max_index = 0;
        for (size_t i = 1; i < final_output.numel(); ++i)
        {
            max_index = final_output[max_index].value() > final_output[i].value() ? max_index : i;
        }
        return max_index;
    };

    std::cout << text << std::flush;
    while (true)
    {
        auto nt = next_token_id(token_ids);
        if (nt == tokenizer.token_to_id("<|endoftext|>") or token_ids.size() > 50)
            break;
        token_ids.push_back(nt);
        std::cout << tokenizer.decode({nt}) << std::flush;
    }
    std::cout << std::endl;

    return 0;
}
