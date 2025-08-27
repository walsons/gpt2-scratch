#ifndef BPE_TOKENIZER_H
#define BPE_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <memory>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <functional>
#include <re2/re2.h>

// Custom hash function for std::pair<std::string, std::string>
struct PairHash 
{
    std::size_t operator()(const std::pair<std::string, std::string>& p) const 
    {
        auto h1 = std::hash<std::string>{}(p.first);
        auto h2 = std::hash<std::string>{}(p.second);
        // Combine the two hash values
        return h1 ^ (h2 << 1);
    }
};

class BPETokenizer 
{
public:
    BPETokenizer(const std::string& vocab_path, 
                 const std::string& merges_path,
                 const std::vector<std::string>& special_tokens = {});

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& token_ids) const;
    std::vector<std::string> tokenize(const std::string& text) const;
    
    int token_to_id(const std::string& token) const;
    std::string id_to_token(int token_id) const;

private:
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> id_to_token_;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks_;
    std::unordered_map<unsigned char, char16_t> byte_encoder_;
    std::unordered_map<char16_t, unsigned char> byte_decoder_;
    std::unordered_map<std::string, int> special_tokens_;
    mutable std::unordered_map<std::string, std::string> cache_;
    
    RE2 pat_;

    void load_vocab(const std::string& path);
    void load_merges(const std::string& path);
    void initialize_byte_encoder_decoder();
    
    std::unordered_set<std::pair<std::string, std::string>, PairHash> get_pairs(const std::vector<std::string>& word) const;
    std::string bpe(const std::string& token) const;
    std::vector<int> encode_part(const std::string& text) const;
    
    std::string convert_bytes(const std::string &bytes) const;
    std::string revert_bytes(const std::string& str) const;
};

#endif // BPE_TOKENIZER_H