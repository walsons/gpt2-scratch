#include "bpe_tokenizer.h"
#include <iostream>

BPETokenizer::BPETokenizer(const std::string& vocab_path, 
                           const std::string& merges_path,
                           const std::vector<std::string>& special_tokens) 
    : pat_(RE2("('s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+\\(?!\\S\\)|\\s+)"))
{
    // Load vocabulary and merges
    load_vocab(vocab_path);
    load_merges(merges_path);
    initialize_byte_encoder_decoder();
    
    // Initialize special tokens
    int next_id = static_cast<int>(vocab_.size());
    for (const auto& token : special_tokens) 
    {
        if (vocab_.find(token) == vocab_.end())
            special_tokens_[token] = next_id++;
        else
            special_tokens_[token] = vocab_[token];  // For special token in vocab
    }
    
    // Build reverse mapping
    for (const auto& pair : vocab_) 
    {
        id_to_token_[pair.second] = pair.first;
    }
    for (const auto& pair : special_tokens_) 
    {
        id_to_token_[pair.second] = pair.first;
    }
}

void BPETokenizer::load_vocab(const std::string& path) 
{
    std::ifstream file(path);
    if (!file.is_open()) 
        throw std::runtime_error("Failed to open vocab file: " + path);
    
    std::string content{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    
    // Simple JSON parsing for vocab.json
    size_t pos = 0;
    std::string key, value;
    while (pos < content.size()) 
    {
        // Find key
        if (content[pos] == '"')
        {
            ++pos;
            while (content[pos] != '"')
            {
                if (content[pos] == '\\')
                    ++pos;
                key.push_back(content[pos]);
                ++pos;
            }
        }

        // Find value
        if (std::isdigit(content[pos]))
        {
            while (std::isdigit(content[pos]))
            {
                value.push_back(content[pos]);
                ++pos;
            }
        }

        if (!key.empty() && !value.empty())
        {
            vocab_[key] = std::stoi(value);
            key.clear();
            value.clear();
        }

        ++pos;
    }
}

void BPETokenizer::load_merges(const std::string& path) 
{
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Failed to open merges file: " + path);
    
    std::string line;
    int rank = 0;
    // skip first line (version info)
    std::getline(file, line);
    while (std::getline(file, line)) 
    {
        if (line.empty())
            continue;
        
        std::istringstream iss(line);
        std::string first, second;
        if (iss >> first >> second) 
        {
            bpe_ranks_[{first, second}] = rank++;
        }
    }
}

void BPETokenizer::initialize_byte_encoder_decoder() 
{
    // Initialize byte encoder following GPT-2's approach
    std::vector<int> bs;
    // Printable ASCII (33-126)
    for (int i = 33; i <= 126; ++i) bs.push_back(i);
    // Latin-1 supplement printable characters (161-172, 174-255)
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    // Create Unicode codepoint mappings (direct for printable chars)
    std::vector<int> cs = bs;

    // For non-printable bytes, map to higher code points starting at 256
    int n = 0;
    for (int b = 0; b < 256; ++b) 
    {
        if (std::find(bs.begin(), bs.end(), b) == bs.end())
        {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    // Build the encoder and decoder mappings (char -> wchar_t) and (wchar_t -> char)
    for (size_t i = 0; i < bs.size(); ++i)
    {
        unsigned char c1 = static_cast<unsigned char>(bs[i]);
        char16_t c2 = static_cast<char16_t>(cs[i]);
        byte_encoder_[c1] = c2;
        byte_decoder_[c2] = c1;
    }
}

std::unordered_set<std::pair<std::string, std::string>, PairHash> 
BPETokenizer::get_pairs(const std::vector<std::string>& word) const 
{
    std::unordered_set<std::pair<std::string, std::string>, PairHash> pairs;
    if (word.size() < 2) 
        return pairs;
    
    std::string prev_char = word[0];
    for (size_t i = 1; i < word.size(); ++i) 
    {
        pairs.insert({prev_char, word[i]});
        prev_char = word[i];
    }
    
    return pairs;
}

std::string BPETokenizer::bpe(const std::string& token) const 
{
    // Check cache
    auto cache_it = cache_.find(token);
    if (cache_it != cache_.end()) 
    {
        return cache_it->second;
    }
    
    std::vector<std::string> word;
    for (size_t i = 0; i < token.size(); ++i)
    {
        std::string w;
        if ((token[i] & 0x80) == 0)
        {
            w.push_back(token[i]);
        }
        else if ((token[i] & 0xE0) == 0xC0)
        {
            w.push_back(static_cast<unsigned char>(token[i]));
            ++i;
            assert(i < token.size());
            w.push_back(static_cast<unsigned char>(token[i]));
        }
        else if ((token[i] & 0xF0) == 0xE0)
        {
            w.push_back(static_cast<unsigned char>(token[i]));
            ++i;
            assert(i < token.size());
            w.push_back(static_cast<unsigned char>(token[i]));
            ++i;
            assert(i < token.size());
            w.push_back(static_cast<unsigned char>(token[i]));
        }
        else if ((token[i] & 0xF8) == 0xF0)
        {
            w.push_back(static_cast<unsigned char>(token[i]));
            ++i;
            assert(i < token.size());
            w.push_back(static_cast<unsigned char>(token[i]));
            ++i;
            assert(i < token.size());
            w.push_back(static_cast<unsigned char>(token[i]));
            ++i;
            assert(i < token.size());
            w.push_back(static_cast<unsigned char>(token[i]));
        }
        word.push_back(w);
    }

    while (true) 
    {
        auto pairs = get_pairs(word);
        if (pairs.empty()) 
            break;
        
        // Find the highest priority bigram
        std::pair<std::string, std::string> bigram;
        int min_rank = std::numeric_limits<int>::max();
        bool found = false;
        
        for (const auto& pair : pairs) 
        {
            auto it = bpe_ranks_.find(pair);
            if (it != bpe_ranks_.end() && it->second < min_rank) 
            {
                min_rank = it->second;
                bigram = pair;
                found = true;
            }
        }
        
        if (!found) 
            break;
        
        // Merge the bigram
        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) 
        {
            auto j = std::find(word.begin() + i, word.end(), bigram.first);
            if (j == word.end()) 
            {
                new_word.insert(new_word.end(), word.begin() + i, word.end());
                break;
            }
            
            // Add elements before the match
            new_word.insert(new_word.end(), word.begin() + i, j);
            i = j - word.begin();
            
            // Check if next element matches second part of bigram
            if (i < word.size() - 1 && word[i + 1] == bigram.second) 
            {
                new_word.push_back(bigram.first + bigram.second);
                i += 2;
            } 
            else 
            {
                new_word.push_back(word[i]);
                i += 1;
            }
        }
        
        word = new_word;
        if (word.size() == 1)
            break;
    }
    
    std::string result;
    for (size_t i = 0; i < word.size(); ++i) 
    {
        if (i > 0) result += " ";
        result += word[i];
    }
    
    // Cache the result
    cache_[token] = result;
    return result;
}

std::string BPETokenizer::convert_bytes(const std::string &bytes) const
{
    std::string result;
    size_t index = 0;
    while (index < bytes.size())
    {
        char16_t unicode_value = std::numeric_limits<char16_t>::max();
        // 1 byte
        if ((bytes[index] & 0x80) == 0x00)
        {
            unicode_value = static_cast<char16_t>(bytes[index]);
        }
        // 2 byte
        else if ((bytes[index] & 0xE0) == 0xC0)
        {
            assert(++index < bytes.size());
            assert((bytes[index] & 0xC0) == 0x80);
            unicode_value = ((bytes[index - 1] & 0x1F) << 6) | (bytes[index] & 0x3F);
        }
        // 3 and 4 bytes don't need to map

        if (unicode_value < 0xFF)
        {
            auto it = byte_encoder_.find(static_cast<unsigned char>(unicode_value));
            if (it != byte_encoder_.end())
            {
                unicode_value = static_cast<char16_t>(it->second);
            }
        }

        if (unicode_value <= 0x7F)
        {
            result.push_back(static_cast<unsigned char>(unicode_value));
        }
        else if (unicode_value <= 0x7FF) 
        {
            result.push_back(static_cast<unsigned char>(0xC0 | ((unicode_value >> 6) & 0x1F)));
            result.push_back(static_cast<char>(0x80 | (unicode_value & 0x3F)));
        } 
        else
        {
            result.push_back(bytes[index]);
        }

        ++index;
    }

    return result;
}

std::string BPETokenizer::revert_bytes(const std::string& str) const
{
    std::string result;
    size_t index = 0;
    size_t len = 1;
    while (index < str.size()) 
    {
        char16_t unicode_value = std::numeric_limits<char16_t>::max();
        // 1 bytes
        if ((str[index] & 0x80) == 0x00) 
        {
            unicode_value = static_cast<char16_t>(str[index]);
            len = 1;
        } 
        // 2 bytes
        else if ((str[index] & 0xE0) == 0xC0) 
        {
            assert(index + 1 < str.size());
            assert((str[index + 1] & 0xC0) == 0x80);
            unicode_value = ((str[index] & 0x1F) << 6 | (str[index + 1] & 0x3F));
            len = 2;
        } 
        // 3 and 4 bytes don't need to map

        auto it = byte_decoder_.find((unicode_value));
        if (it != byte_decoder_.end()) 
        {
            result.push_back(it->second);
        } 
        else 
        {
            result.append(str.substr(index, len));
        }
        index += len;
    }
    return result;
}

std::vector<int> BPETokenizer::encode_part(const std::string& text) const 
{
    std::vector<int> token_ids;

    re2::StringPiece input(text);
    std::string token;
    while (RE2::FindAndConsume(&input, pat_, &token)) 
    {
        // Convert bytes (i.e. " " to "Ġ")
        std::string token_bytes = token;
        std::string token_unicode = convert_bytes(token_bytes);

        // Apply BPE
        std::string bpe_result = bpe(token_unicode);
        
        // Split and convert to IDs
        std::istringstream iss(bpe_result);
        std::string subword;
        while (iss >> subword) 
        {
            auto vocab_it = vocab_.find(subword);
            if (vocab_it != vocab_.end()) 
            {
                token_ids.push_back(vocab_it->second);
            } 
            else 
            {
                throw std::runtime_error("bpe tokenizer encounters unknown token");
            }
        }
    }

    return token_ids;
}

std::vector<int> BPETokenizer::encode(const std::string& text) const 
{
    // Key is position, value is token
    std::vector<std::pair<size_t, std::string>> pos_token;
    for (const auto &[special_token, special_token_id] : special_tokens_)
    {
        size_t pos = text.find(special_token);
        while (pos != std::string::npos)
        {
            pos_token.emplace_back(pos, special_token);
            pos += special_token.length();
            pos = text.find(special_token, pos);
        }
    }

    std::sort(pos_token.begin(), pos_token.end(), [](const auto &a, const auto &b) {
        return a.first < b.first;
    });

    std::vector<int> token_ids;
    size_t start = 0;
    for (const auto &[pos, token] : pos_token)
    {
        // Encode text before special token
        std::string before = text.substr(start, pos - start);
        auto before_tokens = encode_part(before);
        token_ids.insert(token_ids.end(), before_tokens.begin(), before_tokens.end());

        // Add special token
        token_ids.push_back(token_to_id(token));
        
        start = pos + token.length();
    }

    // Encode remaining text
    std::string remaining = text.substr(start);
    auto remaining_tokens = encode_part(remaining);
    token_ids.insert(token_ids.end(), remaining_tokens.begin(), remaining_tokens.end());

    return token_ids;
}

std::string BPETokenizer::decode(const std::vector<int>& token_ids) const 
{
    std::string result;
    
    for (int token_id : token_ids) 
    {
        auto it = id_to_token_.find(token_id);
        if (it != id_to_token_.end()) 
        {
            result += it->second;
        } 
        else 
        {
            throw std::runtime_error("bpe tokenizer encounters unknown token id");
        }
    }
    
    return revert_bytes(result);
}

std::vector<std::string> BPETokenizer::tokenize(const std::string& text) const 
{
    std::vector<int> token_ids = encode(text);
    std::vector<std::string> tokens;
    
    for (int token_id : token_ids) 
    {
        tokens.push_back(decode({token_id}));
    }
    
    return tokens;
}

int BPETokenizer::token_to_id(const std::string& token) const 
{
    auto spec_it = special_tokens_.find(token);
    if (spec_it != special_tokens_.end()) 
    {
        return spec_it->second;
    }
    
    auto vocab_it = vocab_.find(token);
    if (vocab_it != vocab_.end()) 
    {
        return vocab_it->second;
    }
    
    return -1; // Not found
}

std::string BPETokenizer::id_to_token(int token_id) const 
{
    auto it = id_to_token_.find(token_id);
    if (it != id_to_token_.end()) 
    {
        return it->second;
    }
    return "<|unknown|>";
}