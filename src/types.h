#include <vector>
#include <cstdint>
#include <cstring>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include <assert.h>

enum class DataType 
{
    FLOAT_32 = 1,
    FLOAT_64,
    INT_32,
};

using Float32 = float;
using Float64 = double;
using Int32 = int32_t;


// template to implement a map between dtype and dtype enum value
template <typename DType>
struct dtype_enum_value;

template <>
struct dtype_enum_value<Float32> 
{
    static constexpr DataType v = DataType::FLOAT_32;
};

template <>
struct dtype_enum_value<Float64> 
{
    static constexpr DataType v = DataType::FLOAT_64;
};

template <>
struct dtype_enum_value<Int32> 
{
    static constexpr DataType v = DataType::INT_32;
};


using TokenIndex = Int32;
using PosIndex = Int32;


template <typename DType>
class Tensor 
{
public:
    template <typename T> friend std::ostream& operator<<(std::ostream& os, const Tensor<T> &tensor);

    template <typename T> friend Tensor<T> MatMul1DTensor(const Tensor<T> &tensor, const Tensor<T> &other);
    template <typename T> friend Tensor<T> MatMulMatrix(const Tensor<T> &tensor, const Tensor<T> &other);
    template <typename T> friend Tensor<T> MatMulDFS(const Tensor<T> &tensor, const Tensor<T> &other);
    template <typename T> friend Tensor<T> MatMul(Tensor<T> tensor, Tensor<T> other);

    template <typename T> friend Tensor<T> Add1DTensor(const Tensor<T> &tensor, const Tensor<T> &other);
    template <typename T> friend Tensor<T> AddDFS(const Tensor<T> &tensor, const Tensor<T> &other);
    template <typename T> friend Tensor<T> operator+(Tensor<T> tensor, Tensor<T> other);

    template <typename T> friend Tensor<T> Subtract1DTensor(const Tensor<T> &tensor, const Tensor<T> &other);
    template <typename T> friend Tensor<T> SubtractDFS(const Tensor<T> &tensor, const Tensor<T> &other);
    template <typename T> friend Tensor<T> operator-(Tensor<T> tensor, Tensor<T> other);

    template <typename T> friend Tensor<T> Multiply1DTensor(const Tensor<T> &tensor, const Tensor<T> &other);
    template <typename T> friend Tensor<T> MultiplyDFS(const Tensor<T> &tensor, const Tensor<T> &other);
    template <typename T> friend Tensor<T> operator*(Tensor<T> tensor, Tensor<T> other);

    template <typename T> friend Tensor<T> Divide1DTensor(const Tensor<T> &tensor, const Tensor<T> &other);
    template <typename T> friend Tensor<T> DivideDFS(const Tensor<T> &tensor, const Tensor<T> &other);
    template <typename T> friend Tensor<T> operator/(Tensor<T> tensor, Tensor<T> other);

    template <typename T, typename ScalarType, typename> friend Tensor<T> operator+(Tensor<T> tensor, ScalarType scalar);
    template <typename T, typename ScalarType, typename> friend Tensor<T> operator-(Tensor<T> tensor, ScalarType scalar);
    template <typename T, typename ScalarType, typename> friend Tensor<T> operator*(Tensor<T> tensor, ScalarType scalar);
    template <typename T, typename ScalarType, typename> friend Tensor<T> operator/(Tensor<T> tensor, ScalarType scalar);

    template <typename T> friend class Embedding;
    template <typename T> friend class GELU;
    template <typename T> friend class LayerNorm;

    using Dim = int32_t;

    Tensor() = default;
    Tensor(std::vector<Dim> dims_);
    Tensor(std::vector<Dim> dims_, DType* data_, bool transfer_ownership = false);
    Tensor(const Tensor &other);
    Tensor& operator=(const Tensor &other);
    Tensor(Tensor &&other) noexcept;
    Tensor& operator=(Tensor &&other);
    ~Tensor(); 

private:
    size_t numel(const std::vector<Dim> &dims) const ;
    void transpose_dfs(Dim d1, Dim d2, std::vector<Dim> &index, std::vector<Dim> &each_dim_size, std::vector<Dim> &swap_each_dim_size);
    std::string tensor_data_string() const;

public:
    DataType dtype() const { return m_dtype; }
    const std::vector<Dim> &dims() const { return m_dims; }
    size_t numel() const  { return m_data == nullptr ? 0 : numel(m_dims); }
    DType value() const;

    Tensor& operator[](Dim index);
    const Tensor& operator[](Dim index) const;

    Tensor& transpose(Dim d1, Dim d2);
    Tensor& resize(std::vector<Dim> new_dims);
    Tensor& view(std::vector<Dim> new_dims);

    // Softmax operation along the last dimension
    Tensor& softmax();
    // current only support last dimension
    Tensor& mean(bool keep_dim = false);
    // current only support last dimension
    Tensor& var(bool keep_dim = false, bool unbiased = true);

private:
    DataType m_dtype = dtype_enum_value<DType>::v; // Data type of the tensor
    std::vector<Dim> m_dims;
    bool m_data_owner = true; // Indicates if the data is owned by this Tensor
    DType *m_data = nullptr;
    // store sub tensor here for operator[], sub tensor use data from m_data but set m_data_owner to false
    mutable std::unordered_map<Dim, Tensor> m_children; 
    DType *m_transpose_buffer = nullptr; // Used for transpose operation
};


template <typename DType>
Tensor<DType>::Tensor(std::vector<Dim> dims_)
    : m_dtype(dtype_enum_value<DType>::v), m_dims(std::move(dims_)), m_data_owner(true), m_data(new DType[numel(m_dims)])
{
}

template <typename DType>
Tensor<DType>::Tensor(std::vector<Dim> dims_, DType* data_, bool transfer_ownership)
    : m_dtype(dtype_enum_value<DType>::v), m_dims(std::move(dims_)), m_data_owner(true) 
{
    if (data_ != nullptr)
    {
        if (transfer_ownership) 
        {
            m_data = data_;
            data_ = nullptr;
        } 
        else 
        {
            m_data = new DType[numel(m_dims)];
            std::memcpy(m_data, data_, numel(m_dims) * sizeof(DType));
        }
    }
}

// No need to copy m_children
template <typename DType>
Tensor<DType>::Tensor(const Tensor &other)
    : m_dtype(other.m_dtype), m_dims(other.m_dims), m_data_owner(true) 
{
    if (other.m_data != nullptr) 
    {
        m_data = new DType[numel(m_dims)];
        std::memcpy(m_data, other.m_data, numel(m_dims) * sizeof(DType));
    } 
}

// No need to copy m_children, but need to clear current children
template <typename DType>
Tensor<DType>& Tensor<DType>::operator=(const Tensor &other) 
{
    if (this != &other) 
    {
        if (m_data_owner) 
        {
            if (m_data) 
                delete[] static_cast<DType*>(m_data);
            m_children.clear();
            m_dtype = other.m_dtype;
            m_dims = other.m_dims;
            if (other.m_data != nullptr) 
            {
                m_data = new DType[numel(m_dims)];
                std::memcpy(m_data, other.m_data, numel(m_dims) * sizeof(DType));
            }
        } 
        else 
        {
            if (m_dims != other.m_dims) 
                throw std::runtime_error("Cannot assign tensor without ownership with different dimensions");
            for (size_t i = 0; i < numel(); ++i)
                static_cast<DType*>(m_data)[i] = static_cast<const DType*>(other.m_data)[i];
        }
    }
    return *this;
}

template <typename DType>
Tensor<DType>::Tensor(Tensor &&other) noexcept
    : m_dtype(other.m_dtype), m_dims(std::move(other.m_dims)), m_data_owner(true), m_data(other.m_data)
{
    other.m_data = nullptr; // Transfer ownership 
}

template <typename DType>
Tensor<DType>& Tensor<DType>::operator=(Tensor &&other) 
{
    if (this != &other) 
    {
        if (m_data_owner)
        {
            if (m_data)
                delete[] static_cast<DType*>(m_data);
            m_children.clear();
            m_dtype = other.m_dtype;
            m_dims = std::move(other.m_dims);
            m_data = other.m_data;
            other.m_data = nullptr; // Transfer ownership
        } 
        else 
        {
            if (m_dims != other.m_dims)
                throw std::runtime_error("Cannot move tensor without ownership with different dimensions");
            for (size_t i = 0; i < numel(); ++i)
                static_cast<DType*>(m_data)[i] = static_cast<DType*>(other.m_data)[i];
        }
    }
    return *this;
}

template <typename DType>
Tensor<DType>::~Tensor() 
{
    if (m_data && m_data_owner)
        delete[] static_cast<DType*>(m_data);
}

template <typename DType>
size_t Tensor<DType>::numel(const std::vector<Dim> &dims) const 
{
    size_t num = 1;
    for (const auto &dim : dims)
        num *= dim;
    return num;
}

template <typename DType>
void Tensor<DType>::transpose_dfs(Dim d1, Dim d2, std::vector<Dim> &index, std::vector<Dim> &each_dim_size, std::vector<Dim> &swap_each_dim_size) 
{
    if (index.size() == m_dims.size()) 
    {
        auto swap_index = index;
        std::swap(swap_index[d1], swap_index[d2]);
        size_t cur_data_index = 0, swap_data_index = 0;
        for (size_t i = 0; i < index.size(); ++i) 
        {
            cur_data_index += index[i] * each_dim_size[i];
            swap_data_index += swap_index[i] * swap_each_dim_size[i];
        }
        static_cast<DType*>(m_transpose_buffer)[swap_data_index] = static_cast<DType*>(m_data)[cur_data_index];
        return;
    }
    for (Dim i = 0; i < m_dims[index.size()]; ++i) 
    {
        index.push_back(i);
        transpose_dfs(d1, d2, index, each_dim_size, swap_each_dim_size);
        index.pop_back();
    }
}

template <typename DType>
std::string Tensor<DType>::tensor_data_string() const
{
    auto &tensor = *this;
    std::string str;
    if (tensor.m_dims.empty()) 
    {
        if (tensor.m_data == nullptr) 
            return "None";

        std::ostringstream oss;
        oss << std::scientific << std::setprecision(4);
        oss << *static_cast<DType*>(tensor.m_data);
        return oss.str();
    } 
    else if (tensor.m_dims.size() == 1) 
    {
        str += "[";
        if (tensor.m_dims[0] < 10) 
        {
            for (typename Tensor<DType>::Dim i = 0; i < tensor.m_dims[0]; ++i)
                str += tensor[i].tensor_data_string() + ", ";
        }
        else // if the first dimension is large, we only show the first 3 elements and the last 3 elements
        {
            for (typename Tensor<DType>::Dim i = 0; i < 3; ++i)
                str += tensor[i].tensor_data_string() + ", ";
            str += "..., ";
            for (typename Tensor<DType>::Dim i = tensor.m_dims[0] - 3; i < tensor.m_dims[0]; ++i)
                str += tensor[i].tensor_data_string() + ", ";
        }
        // remove last comma and space and add closing bracket
        str.pop_back();
        str.back() = ']';
    } 
    else 
    {
        str += "[";
        if (tensor.m_dims[0] < 10) 
        {
            for (typename Tensor<DType>::Dim i = 0; i < tensor.m_dims[0]; ++i)
                str += tensor[i].tensor_data_string() + "," + std::string(tensor.m_dims.size() - 1, '\n');
        } 
        else // if the first dimension is large, we only show the first 3 elements and the last 3 elements
        {
            for (typename Tensor<DType>::Dim i = 0; i < 3; ++i)
                str += tensor[i].tensor_data_string() + "," + std::string(tensor.m_dims.size() - 1, '\n');
            str += "...,\n";
            for (typename Tensor<DType>::Dim i = tensor.m_dims[0] - 3; i < tensor.m_dims[0]; ++i)
                str += tensor[i].tensor_data_string() + "," + std::string(tensor.m_dims.size() - 1, '\n');
        }
        // remove last ",\n\n..." and add closing bracket
        for (size_t i = 0; i < tensor.m_dims.size() - 1; ++i)
            str.pop_back();
        str.back() = ']';
    }
    return str;
}

template <typename DType>
DType Tensor<DType>::value() const 
{
    if (m_data == nullptr || !m_dims.empty())
        throw std::runtime_error("Only scalar tensors have a value");

    return *m_data;
}

template <typename DType>
Tensor<DType>& Tensor<DType>::operator[](Dim index) 
{
    if (m_dims.empty())
        throw std::runtime_error("Tensor has no dimensions");
    
    if (index < 0 || index >= m_dims[0])
        throw std::out_of_range("Index out of range");

    if (m_children.find(index) != m_children.end())
        return m_children.at(index);

    m_children[index] = {};
    m_children[index].m_dtype = m_dtype;
    m_children[index].m_dims = std::vector<Dim>(m_dims.begin() + 1, m_dims.end());
    m_children[index].m_data = static_cast<DType*>(m_data) + index * numel() / m_dims[0];
    m_children[index].m_data_owner = false;
    return m_children[index];
}

template <typename DType>
const Tensor<DType>& Tensor<DType>::operator[](Dim index) const
{
    if (m_dims.empty())
        throw std::runtime_error("Tensor has no dimensions");
    
    if (index < 0 || index >= m_dims[0])
        throw std::out_of_range("Index out of range");

    if (m_children.find(index) != m_children.end())
        return m_children.at(index);

    m_children[index] = {};
    m_children[index].m_dtype = m_dtype;
    m_children[index].m_dims = std::vector<Dim>(m_dims.begin() + 1, m_dims.end());
    m_children[index].m_data = static_cast<DType*>(m_data) + index * numel() / m_dims[0];
    m_children[index].m_data_owner = false;
    return m_children[index];
}

template <typename DType>
Tensor<DType>& Tensor<DType>::transpose(Dim d1, Dim d2) 
{
    if (d1 < 0 || d1 >= static_cast<Dim>(m_dims.size()) || d2 < 0 || d2 >= static_cast<Dim>(m_dims.size()))
        throw std::out_of_range("Dimension index out of range");

    if (!m_data_owner)
        throw std::runtime_error("Cannot transpose tensor without ownership");

    if (d1 == d2)
        return *this; // No need to transpose if the dimensions are the same

    m_transpose_buffer = new DType[numel(m_dims)];
    std::memcmp(m_transpose_buffer, m_data, numel(m_dims) * sizeof(DType));

    // prepare two vectors to calculate the index
    auto swap_dims = m_dims;
    std::swap(swap_dims[d1], swap_dims[d2]);
    std::vector<Dim> each_dim_size{1}, swap_each_dim_size{1};
    for (auto it = m_dims.rbegin(); it != m_dims.rend() - 1; ++it)
        each_dim_size.push_back(*it * each_dim_size.back());
    for (auto it = swap_dims.rbegin(); it != swap_dims.rend() - 1; ++it)
        swap_each_dim_size.push_back(*it * swap_each_dim_size.back());
    std::reverse(each_dim_size.begin(), each_dim_size.end());
    std::reverse(swap_each_dim_size.begin(), swap_each_dim_size.end());

    std::vector<Dim> index{};
    transpose_dfs(d1, d2, index, each_dim_size, swap_each_dim_size);
    // view to new dimensions
    std::swap(m_dims[d1], m_dims[d2]);
    if (m_data)
        delete[] static_cast<DType*>(m_data);
    m_data = m_transpose_buffer;
    m_transpose_buffer = nullptr; // Clear the buffer after use
    return *this;
}

template <typename DType>
Tensor<DType>& Tensor<DType>::resize(std::vector<Dim> new_dims)
{
    if (!m_data_owner)
        throw std::runtime_error("Cannot resize tensor without ownership");

    DType* new_data = new DType[numel(new_dims)];
    std::memcpy(new_data, m_data, std::min(numel(new_dims), numel()) * sizeof(DType));
    if (m_data)
        delete[] static_cast<DType*>(m_data);
    m_data = new_data;
    m_dims = std::move(new_dims);
    return *this;
}

template <typename DType>
Tensor<DType>& Tensor<DType>::view(std::vector<Dim> new_dims) 
{
    if (numel(new_dims) != numel(m_dims))
        throw std::runtime_error("Cannot view tensor with different number of elements");

    if (!m_data_owner)
        throw std::runtime_error("Cannot view tensor without ownership");

    m_dims = std::move(new_dims);
    return *this;
}

template <typename DType>
Tensor<DType>& Tensor<DType>::softmax()
{
    if (m_dims.empty())
        return *this;

    if (m_dims.size() == 1)
    {
        DType max_val = *std::max_element(m_data, m_data + m_dims[0]);
        std::for_each(m_data, m_data + m_dims[0], [&](DType &val) {
            if (std::isnan(val) || std::isinf(val)) 
                val = 0; // Handle NaN and Inf values: -std::numeric_limits<DType>::infinity()
            else
                val = std::exp(val - max_val);
        });
        DType sum = std::accumulate(m_data, m_data + m_dims[0], static_cast<DType>(0));
        if (std::isnan(sum) || std::isinf(sum) || sum == 0)
            std::cout << "Warning: Softmax sum is NaN, Inf or zero" << std::endl;
        std::for_each(m_data, m_data + m_dims[0], [sum](DType &val) {
            val /= sum;
        });
    }
    
    for (Dim i = 0; i < m_dims[0]; ++i) 
    {
        (*this)[i].softmax();
    }
    return *this;
}

template <typename DType>
Tensor<DType>& Tensor<DType>::mean(bool keep_dim)
{
    if (m_dims.empty())
        return *this;

    std::vector<Dim> new_dims = m_dims;
    new_dims.erase(new_dims.end() - 1);
    Tensor<DType> result(new_dims);

    for (size_t i = 0; i < result.numel(); ++i) 
    {
        result.m_data[i] = 0;
        for (size_t j = i * m_dims.back(); j < (i + 1)* m_dims.back(); ++j) 
        {
            result.m_data[i] += m_data[j];
        }
    }

    for (size_t i = 0; i < result.numel(); ++i) 
    {
        result.m_data[i] /= m_dims.back();
    }

    if (keep_dim) 
    {
        result.m_dims.insert(result.m_dims.end(), 1);
    }

    *this = std::move(result);
    return *this;
}

template <typename DType>
Tensor<DType>& Tensor<DType>::var(bool keep_dim, bool unbiased)
{
    if (m_dims.empty())
        return *this;

    std::vector<Dim> new_dims = m_dims;
    new_dims.erase(new_dims.end() - 1);
    Tensor<DType> result(new_dims);

    for (size_t i = 0; i < result.numel(); ++i) 
    {
        DType mean_val = 0;
        for (size_t j = i * m_dims.back(); j < (i + 1) * m_dims.back(); ++j) 
        {
            mean_val += m_data[j];
        }
        mean_val /= m_dims.back();

        result.m_data[i] = 0;
        for (size_t j = i * m_dims.back(); j < (i + 1) * m_dims.back(); ++j) 
        {
            result.m_data[i] += (m_data[j] - mean_val) * (m_data[j] - mean_val);
        }
        if (unbiased && m_dims.back() > 1) 
        {
            result.m_data[i] /= (m_dims.back() - 1);
        } 
        else 
        {
            result.m_data[i] /= m_dims.back();
        }
    }

    if (keep_dim) 
    {
        result.m_dims.insert(result.m_dims.end(), 1);
    }

    *this = std::move(result);
    return *this;
}

template <typename DType>
std::ostream& operator<<(std::ostream& os, const Tensor<DType> &tensor) 
{
    os << "Tensor(";
    os << tensor.tensor_data_string();
    os << ", dtype=";
    switch (tensor.dtype()) 
    {
        case DataType::FLOAT_32:
            os << "DType";
            break;
        case DataType::FLOAT_64:
            os << "FLoat64";
            break;
        default:
            os << "Unknown";
            break;
    }
    os << ", dims=[";
    for (size_t i = 0; i < tensor.dims().size(); ++i) 
    {
        os << tensor.dims()[i];
        if (i + 1 < tensor.dims().size())
            os << ", ";
    }
    os << "])" << std::endl;
    return os;
}

template <typename DType>
Tensor<DType> MatMul1DTensor(const Tensor<DType> &tensor, const Tensor<DType> &other) 
{
    assert(tensor.m_dims.size() == 1 && other.m_dims.size() == 1);
    assert(tensor.m_dims[0] == other.m_dims[0]);
    Tensor<DType> result;
    result.resize(std::vector<typename Tensor<DType>::Dim>{1});
    DType num = 0;
    for (typename Tensor<DType>::Dim i = 0; i < tensor.m_dims[0]; ++i) 
    {
        num += static_cast<DType*>(tensor.m_data)[i] * static_cast<DType*>(other.m_data)[i];
    }
    static_cast<DType*>(result.m_data)[0] = num;
    return result[0];  // remove the extra dimension
}

template <typename DType>
Tensor<DType> MatMulMatrix(const Tensor<DType> &tensor, const Tensor<DType> &other) 
{
    assert(tensor.m_dims.size() == 2 && other.m_dims.size() == 2);
    assert(tensor.m_dims[1] == other.m_dims[0]);
    Tensor<DType> result;
    result.resize(std::vector<typename Tensor<DType>::Dim>{tensor.m_dims[0], other.m_dims[1]});
    Tensor<DType> transposed_other = other;
    transposed_other.transpose(0, 1); // Transpose the second tensor for easier multiplication
    for (typename Tensor<DType>::Dim i = 0; i < result.m_dims[0]; ++i) 
    {
        for (typename Tensor<DType>::Dim j = 0; j < result.m_dims[1]; ++j) 
        {
            Tensor num = MatMul1DTensor(tensor[i], transposed_other[j]);
            result[i][j] = std::move(num);
        }
    }
    return result;
}

template <typename DType>
Tensor<DType> MatMulDFS(const Tensor<DType> &tensor, const Tensor<DType> &other) 
{
    assert(tensor.m_dims.size() == other.m_dims.size() && tensor.m_dims.size() >= 2);
    if (tensor.m_dims.size() == 2)
        return MatMulMatrix(tensor, other);

    Tensor<DType> result;
    auto dim = std::max(tensor.m_dims[0], other.m_dims[0]);
    for (typename Tensor<DType>::Dim i = 0; i < dim; ++i) 
    {
        typename Tensor<DType>::Dim tensor_index = i, other_index = i;
        if (tensor.m_dims[0] == 1)
            tensor_index = 0;
        if (other.m_dims[0] == 1)
            other_index = 0;
        auto sub_result = MatMulDFS(tensor[tensor_index], other[other_index]);
        if (i == 0) 
        {
            auto result_dims = sub_result.m_dims;
            result_dims.insert(result_dims.begin(), std::max(tensor.m_dims[0], other.m_dims[0]));
            result.resize(result_dims);
        }
        result[i] = std::move(sub_result);
    }
    return result;
}

template <typename DType>
Tensor<DType> MatMul(Tensor<DType> tensor, Tensor<DType> other) 
{
    if (tensor.m_dims.size() < 2 || other.m_dims.size() < 2) 
    {
        // support 1D tensor in the future
        throw std::runtime_error("Both tensors must have at least 2 dimensions for matrix multiplication");
    }

    if (tensor.m_dims[tensor.m_dims.size() - 1] != other.m_dims[other.m_dims.size() - 2])
        throw std::runtime_error("The last dimension of the first tensor must match the second to last dimension of the second tensor");

    // broadcast the tensors
    if (tensor.m_dims.size() > other.m_dims.size()) 
    {
        auto append = decltype(tensor.m_dims)(tensor.m_dims.size() - other.m_dims.size(), 1);
        other.m_dims.insert(other.m_dims.begin(), append.begin(), append.end());
    }
    else if (tensor.m_dims.size() < other.m_dims.size()) 
    {
        auto append = decltype(other.m_dims)(other.m_dims.size() - tensor.m_dims.size(), 1);
        tensor.m_dims.insert(tensor.m_dims.begin(), append.begin(), append.end());
    }
    for (size_t i = 0; i < tensor.m_dims.size() - 2; ++i) 
    {
        if (tensor.m_dims[i] != other.m_dims[i]) 
        {
            if (tensor.m_dims[i] != 1 && other.m_dims[i] != 1)
                throw std::runtime_error("The dimensions of the tensors must match or be 1 for broadcasting");
        }
    }
    return MatMulDFS(tensor, other);
}

template <typename DType>
Tensor<DType> Add1DTensor(const Tensor<DType> &tensor, const Tensor<DType> &other) 
{
    assert(tensor.m_dims.size() == 1 && other.m_dims.size() == 1);

    Tensor<DType> result;
    typename Tensor<DType>::Dim result_dim = std::max(tensor.m_dims[0], other.m_dims[0]);
    result.resize({result_dim});
    for (typename Tensor<DType>::Dim i = 0; i < result_dim; ++i) 
    {
        typename Tensor<DType>::Dim tensor_index = i, other_index = i;
        if (tensor.m_dims[0] == 1)
            tensor_index = 0;
        if (other.m_dims[0] == 1)
            other_index = 0;
        static_cast<DType*>(result.m_data)[i] = static_cast<const DType*>(tensor.m_data)[tensor_index] + static_cast<const DType*>(other.m_data)[other_index];
    }
    return result;
}

template <typename DType>
Tensor<DType> AddDFS(const Tensor<DType> &tensor, const Tensor<DType> &other) 
{
    assert(tensor.m_dims.size() == other.m_dims.size());
    if (tensor.m_dims.size() == 1)
        return Add1DTensor(tensor, other);

    Tensor<DType> result;
    auto dim = std::max(tensor.m_dims[0], other.m_dims[0]);
    for (typename Tensor<DType>::Dim i = 0; i < dim; ++i) 
    {
        typename Tensor<DType>::Dim tensor_index = i, other_index = i;
        if (tensor.m_dims[0] == 1)
            tensor_index = 0;
        if (other.m_dims[0] == 1)
            other_index = 0;
        auto sub_result = AddDFS(tensor[tensor_index], other[other_index]);
        if (i == 0) 
        {
            auto result_dims = sub_result.m_dims;
            result_dims.insert(result_dims.begin(), std::max(tensor.m_dims[0], other.m_dims[0]));
            result.resize(result_dims);
        }
        result[i] = std::move(sub_result);
    }
    return result;
}

template <typename DType>
Tensor<DType> operator+(Tensor<DType> tensor, Tensor<DType> other) 
{
    if (tensor.m_dims.size() < 1 || other.m_dims.size() < 1) 
        throw std::runtime_error("Both tensors must have at least 1 dimension for addition");

    // broadcast the tensors
    if (tensor.m_dims.size() > other.m_dims.size()) 
    {
        auto append = decltype(tensor.m_dims)(tensor.m_dims.size() - other.m_dims.size(), 1);
        other.m_dims.insert(other.m_dims.begin(), append.begin(), append.end());
    } 
    else if (tensor.m_dims.size() < other.m_dims.size()) 
    {
        auto append = decltype(other.m_dims)(other.m_dims.size() - tensor.m_dims.size(), 1);
        tensor.m_dims.insert(tensor.m_dims.begin(), append.begin(), append.end());
    }
    for (size_t i = 0; i < tensor.m_dims.size(); ++i) 
    {
        if (tensor.m_dims[i] != other.m_dims[i]) 
        {
            if (tensor.m_dims[i] != 1 && other.m_dims[i] != 1)
                throw std::runtime_error("The dimensions of the tensors must match or be 1 for broadcasting");
        }
    }
    return AddDFS(tensor, other);
}

template <typename DType>
Tensor<DType> Subtract1DTensor(const Tensor<DType> &tensor, const Tensor<DType> &other) 
{
    assert(tensor.m_dims.size() == 1 && other.m_dims.size() == 1);

    Tensor<DType> result;
    typename Tensor<DType>::Dim result_dim = std::max(tensor.m_dims[0], other.m_dims[0]);
    result.resize({result_dim});
    for (typename Tensor<DType>::Dim i = 0; i < result_dim; ++i) 
    {
        typename Tensor<DType>::Dim tensor_index = i, other_index = i;
        if (tensor.m_dims[0] == 1)
            tensor_index = 0;
        if (other.m_dims[0] == 1)
            other_index = 0;
        static_cast<DType*>(result.m_data)[i] = static_cast<const DType*>(tensor.m_data)[tensor_index] - static_cast<const DType*>(other.m_data)[other_index];
    }
    return result;
}

template <typename DType>
Tensor<DType> SubtractDFS(const Tensor<DType> &tensor, const Tensor<DType> &other) 
{
    assert(tensor.m_dims.size() == other.m_dims.size());
    if (tensor.m_dims.size() == 1)
        return Subtract1DTensor(tensor, other);

    Tensor<DType> result;
    auto dim = std::max(tensor.m_dims[0], other.m_dims[0]);
    for (typename Tensor<DType>::Dim i = 0; i < dim; ++i) 
    {
        typename Tensor<DType>::Dim tensor_index = i, other_index = i;
        if (tensor.m_dims[0] == 1)
            tensor_index = 0;
        if (other.m_dims[0] == 1)
            other_index = 0;
        auto sub_result = SubtractDFS(tensor[tensor_index], other[other_index]);
        if (i == 0) 
        {
            auto result_dims = sub_result.m_dims;
            result_dims.insert(result_dims.begin(), std::max(tensor.m_dims[0], other.m_dims[0]));
            result.resize(result_dims);
        }
        result[i] = std::move(sub_result);
    }
    return result;
}

template <typename DType>
Tensor<DType> operator-(Tensor<DType> tensor, Tensor<DType> other)
{
    if (tensor.m_dims.size() < 1 || other.m_dims.size() < 1) 
        throw std::runtime_error("Both tensors must have at least 1 dimension for subtraction");

    // broadcast the tensors
    if (tensor.m_dims.size() > other.m_dims.size()) 
    {
        auto append = decltype(tensor.m_dims)(tensor.m_dims.size() - other.m_dims.size(), 1);
        other.m_dims.insert(other.m_dims.begin(), append.begin(), append.end());
    } 
    else if (tensor.m_dims.size() < other.m_dims.size()) 
    {
        auto append = decltype(other.m_dims)(other.m_dims.size() - tensor.m_dims.size(), 1);
        tensor.m_dims.insert(tensor.m_dims.begin(), append.begin(), append.end());
    }
    for (size_t i = 0; i < tensor.m_dims.size(); ++i) 
    {
        if (tensor.m_dims[i] != other.m_dims[i]) 
        {
            if (tensor.m_dims[i] != 1 && other.m_dims[i] != 1)
                throw std::runtime_error("The dimensions of the tensors must match or be 1 for broadcasting");
        }
    }
    return SubtractDFS(tensor, other);
}

template <typename DType>
Tensor<DType> Multiply1DTensor(const Tensor<DType> &tensor, const Tensor<DType> &other) 
{
    assert(tensor.m_dims.size() == 1 && other.m_dims.size() == 1);

    Tensor<DType> result;
    typename Tensor<DType>::Dim result_dim = std::max(tensor.m_dims[0], other.m_dims[0]);
    result.resize({result_dim});
    for (typename Tensor<DType>::Dim i = 0; i < result_dim; ++i) 
    {
        typename Tensor<DType>::Dim tensor_index = i, other_index = i;
        if (tensor.m_dims[0] == 1)
            tensor_index = 0;
        if (other.m_dims[0] == 1)
            other_index = 0;
        static_cast<DType*>(result.m_data)[i] = static_cast<const DType*>(tensor.m_data)[tensor_index] * static_cast<const DType*>(other.m_data)[other_index];
    }
    return result;
}

template <typename DType>
Tensor<DType> MultiplyDFS(const Tensor<DType> &tensor, const Tensor<DType> &other) 
{
    assert(tensor.m_dims.size() == other.m_dims.size());
    if (tensor.m_dims.size() == 1)
        return Multiply1DTensor(tensor, other);

    Tensor<DType> result;
    auto dim = std::max(tensor.m_dims[0], other.m_dims[0]);
    for (typename Tensor<DType>::Dim i = 0; i < dim; ++i) 
    {
        typename Tensor<DType>::Dim tensor_index = i, other_index = i;
        if (tensor.m_dims[0] == 1)
            tensor_index = 0;
        if (other.m_dims[0] == 1)
            other_index = 0;
        auto sub_result = MultiplyDFS(tensor[tensor_index], other[other_index]);
        if (i == 0) 
        {
            auto result_dims = sub_result.m_dims;
            result_dims.insert(result_dims.begin(), std::max(tensor.m_dims[0], other.m_dims[0]));
            result.resize(result_dims);
        }
        result[i] = std::move(sub_result);
    }
    return result;
}

template <typename DType>
Tensor<DType> operator*(Tensor<DType> tensor, Tensor<DType> other) 
{
    if (tensor.m_dims.size() < 1 || other.m_dims.size() < 1) 
        throw std::runtime_error("Both tensors must have at least 1 dimension for multiplication");

    // broadcast the tensors
    if (tensor.m_dims.size() > other.m_dims.size()) 
    {
        auto append = decltype(tensor.m_dims)(tensor.m_dims.size() - other.m_dims.size(), 1);
        other.m_dims.insert(other.m_dims.begin(), append.begin(), append.end());
    } 
    else if (tensor.m_dims.size() < other.m_dims.size()) 
    {
        auto append = decltype(other.m_dims)(other.m_dims.size() - tensor.m_dims.size(), 1);
        tensor.m_dims.insert(tensor.m_dims.begin(), append.begin(), append.end());
    }
    for (size_t i = 0; i < tensor.m_dims.size(); ++i) 
    {
        if (tensor.m_dims[i] != other.m_dims[i]) 
        {
            if (tensor.m_dims[i] != 1 && other.m_dims[i] != 1)
                throw std::runtime_error("The dimensions of the tensors must match or be 1 for broadcasting");
        }
    }
    return MultiplyDFS(tensor, other);
}

template <typename DType>
Tensor<DType> Divide1DTensor(const Tensor<DType> &tensor, const Tensor<DType> &other) 
{
    assert(tensor.m_dims.size() == 1 && other.m_dims.size() == 1);

    Tensor<DType> result;
    typename Tensor<DType>::Dim result_dim = std::max(tensor.m_dims[0], other.m_dims[0]);
    result.resize({result_dim});
    for (typename Tensor<DType>::Dim i = 0; i < result_dim; ++i) 
    {
        typename Tensor<DType>::Dim tensor_index = i, other_index = i;
        if (tensor.m_dims[0] == 1)
            tensor_index = 0;
        if (other.m_dims[0] == 1)
            other_index = 0;
        if (static_cast<const DType*>(other.m_data)[other_index] == 0)
            throw std::runtime_error("Cannot divide by zero");
        static_cast<DType*>(result.m_data)[i] = static_cast<const DType*>(tensor.m_data)[tensor_index] / static_cast<const DType*>(other.m_data)[other_index];
    }
    return result;
}

template <typename DType>
Tensor<DType> DivideDFS(const Tensor<DType> &tensor, const Tensor<DType> &other) 
{
    assert(tensor.m_dims.size() == other.m_dims.size());
    if (tensor.m_dims.size() == 1)
        return Divide1DTensor(tensor, other);

    Tensor<DType> result;
    auto dim = std::max(tensor.m_dims[0], other.m_dims[0]);
    for (typename Tensor<DType>::Dim i = 0; i < dim; ++i) 
    {
        typename Tensor<DType>::Dim tensor_index = i, other_index = i;
        if (tensor.m_dims[0] == 1)
            tensor_index = 0;
        if (other.m_dims[0] == 1)
            other_index = 0;
        auto sub_result = DivideDFS(tensor[tensor_index], other[other_index]);
        if (i == 0) 
        {
            auto result_dims = sub_result.m_dims;
            result_dims.insert(result_dims.begin(), std::max(tensor.m_dims[0], other.m_dims[0]));
            result.resize(result_dims);
        }
        result[i] = std::move(sub_result);
    }
    return result;
}

template <typename DType>
Tensor<DType> operator/(Tensor<DType> tensor, Tensor<DType> other) 
{
    if (tensor.m_dims.size() < 1 || other.m_dims.size() < 1) 
        throw std::runtime_error("Both tensors must have at least 1 dimension for division");

    // broadcast the tensors
    if (tensor.m_dims.size() > other.m_dims.size()) 
    {
        auto append = decltype(tensor.m_dims)(tensor.m_dims.size() - other.m_dims.size(), 1);
        other.m_dims.insert(other.m_dims.begin(), append.begin(), append.end());
    } 
    else if (tensor.m_dims.size() < other.m_dims.size()) 
    {
        auto append = decltype(other.m_dims)(other.m_dims.size() - tensor.m_dims.size(), 1);
        tensor.m_dims.insert(tensor.m_dims.begin(), append.begin(), append.end());
    }
    for (size_t i = 0; i < tensor.m_dims.size(); ++i) 
    {
        if (tensor.m_dims[i] != other.m_dims[i]) 
        {
            if (tensor.m_dims[i] != 1 && other.m_dims[i] != 1)
                throw std::runtime_error("The dimensions of the tensors must match or be 1 for broadcasting");
        }
    }
    return DivideDFS(tensor, other);
}

template <typename DType, typename ScalarType, 
          typename = std::enable_if_t<std::is_convertible_v<ScalarType, DType>>>
Tensor<DType> operator+(Tensor<DType> tensor, ScalarType scalar)
{
    if (tensor.m_data == nullptr)
        throw std::runtime_error("Cannot add a scalar to a tensor with no data");

    Tensor<DType> result = tensor;
    for (size_t i = 0; i < result.numel(); ++i) 
    {
        static_cast<DType*>(result.m_data)[i] += static_cast<DType>(scalar);
    }
    return result;
}

template <typename DType, typename ScalarType, 
          typename = std::enable_if_t<std::is_convertible_v<ScalarType, DType>>>
Tensor<DType> operator+(ScalarType scalar, Tensor<DType> tensor)
{
    return tensor + scalar; // Use the existing operator+ for consistency
}

template <typename DType, typename ScalarType, 
          typename = std::enable_if_t<std::is_convertible_v<ScalarType, DType>>>
Tensor<DType> operator-(Tensor<DType> tensor, ScalarType scalar)
{
    if (tensor.m_data == nullptr)
        throw std::runtime_error("Cannot subtract a scalar from a tensor with no data");

    Tensor<DType> result = tensor;
    for (size_t i = 0; i < result.numel(); ++i) 
    {
        static_cast<DType*>(result.m_data)[i] -= static_cast<DType>(scalar);
    }
    return result;
}

template <typename DType, typename ScalarType, 
          typename = std::enable_if_t<std::is_convertible_v<ScalarType, DType>>>
Tensor<DType> operator*(Tensor<DType> tensor, ScalarType scalar) 
{
    if (tensor.m_data == nullptr)
        throw std::runtime_error("Cannot multiply a tensor with no data");
    
    Tensor<DType> result = tensor;
    for (size_t i = 0; i < result.numel(); ++i) 
    {
        static_cast<DType*>(result.m_data)[i] *= static_cast<DType>(scalar);
    }
    return result;
}

template <typename DType, typename ScalarType, 
          typename = std::enable_if_t<std::is_convertible_v<ScalarType, DType>>>
Tensor<DType> operator*(ScalarType scalar, Tensor<DType> tensor)
{
    return tensor * scalar; // Use the existing operator* for consistency
}

template <typename DType, typename ScalarType, 
          typename = std::enable_if_t<std::is_convertible_v<ScalarType, DType>>>
Tensor<DType> operator/(Tensor<DType> tensor, ScalarType scalar) 
{
    if (tensor.m_data == nullptr)
        throw std::runtime_error("Cannot divide a tensor with no data");
    
    if (scalar == 0)
        throw std::runtime_error("Cannot divide by zero");

    Tensor<DType> result = tensor;
    for (size_t i = 0; i < result.numel(); ++i) 
    {
        static_cast<DType*>(result.m_data)[i] /= static_cast<DType>(scalar);
    }
    return result;
}


// Define types for reading parameters
using ParamKeyLen = int32_t;
using ParamDType = int32_t;
using ParamShapeLen = int32_t;
using ParamShapeElement = int32_t;


/***** Model related types *****/
template <typename DType>
class Embedding
{
public:
    using Dim = typename Tensor<DType>::Dim;

    Embedding(Dim num_embeddings_, Dim embedding_dim_)
        : weight(Tensor<DType>({num_embeddings_, embedding_dim_})) {}
    
    template <typename Index, typename = std::enable_if_t<std::is_integral_v<Index>>>
    Tensor<DType> operator()(const Tensor<Index> &x) const
    {
        assert(x.m_dims.size() > 0);

        Tensor<DType> output;
        auto output_size = x.m_dims;
        output_size.push_back(weight.m_dims[1]);
        output.resize(output_size);
        dfs(x, output);
        return output;
    }

public:
    Tensor<DType> weight;

private:
    template <typename Index, typename = std::enable_if_t<std::is_integral_v<Index>>>
    void dfs(const Tensor<Index> &x, Tensor<DType> &output) const
    {
        if (x.m_dims.size() == 1) 
        {
            for (Dim i = 0; i < x.m_dims[0]; ++i) 
            {
                output[i] = weight[x[i].value()];
            }
            return;
        }
        for (Dim i = 0; i < x.m_dims[0]; ++i) 
        {
            dfs(x[i], output[i]);
        }
    }
};


template <typename DType>
class Linear 
{
public:
    using Dim = typename Tensor<DType>::Dim;

    Linear(Dim in_features_, Dim out_features_, bool bias_ = true)
        : weight(Tensor<DType>({out_features_, in_features_})), 
          bias(bias_ ? Tensor<DType>({out_features_}) : Tensor<DType>()) {}

    Tensor<DType> operator()(const Tensor<DType> &input) const
    {
        auto weight_T = weight;
        Tensor<DType> output = MatMul(input, weight_T.transpose(0, 1));
        if (bias.numel() > 0) 
        {
            return output + bias;
        }
        return output;
    }

public:
    Tensor<DType> weight;
    Tensor<DType> bias;
};


template<typename DType>
class MultiHeadAttention
{
public:
    using Dim = typename Tensor<DType>::Dim;
    MultiHeadAttention(Dim d_in_, Dim d_out_, int context_length_, int num_heads_, bool qkv_bias_ = false)
        : m_d_out(d_out_), m_num_heads(num_heads_), 
          m_head_dim(d_out_ / num_heads_),
          m_W_query(d_in_, d_out_, qkv_bias_),
          m_W_key(d_in_, d_out_, qkv_bias_),
          m_W_value(d_in_, d_out_, qkv_bias_),
          m_out_proj(d_out_, d_out_)
    {
        // Initialize mask for attention
        DType *mask_data = new DType[context_length_ * context_length_];
        for (int i = 0; i < context_length_; ++i) 
        {
            for (int j = 0; j < context_length_; ++j) 
            {
                mask_data[i * context_length_ + j] = (i < j) ? 1 : 0;
            }
        }
        m_mask = Tensor<DType>({context_length_, context_length_}, mask_data, true);
    }

    Tensor<DType> operator()(const Tensor<DType> &x) const
    {
        // Implement the forward pass of multi-head attention
        Dim batch = x.dims()[0];
        Dim num_tokens = x.dims()[1];
        // Dim d_in = x.dims()[2]; unused

        Tensor<DType> keys = m_W_key(x);
        Tensor<DType> queries = m_W_query(x);
        Tensor<DType> values = m_W_value(x);

        keys.view({batch, num_tokens, m_num_heads, m_head_dim});
        queries.view({batch, num_tokens, m_num_heads, m_head_dim});
        values.view({batch, num_tokens, m_num_heads, m_head_dim});

        keys.transpose(1, 2); // Shape: [batch, num_heads, num_tokens, head_dim]
        queries.transpose(1, 2); // Shape: [batch, num_heads, num_tokens, head_dim]
        values.transpose(1, 2); // Shape: [batch, num_heads, num_tokens, head_dim]

        auto attn_scores = MatMul(queries, keys.transpose(2, 3)); // Shape: [batch, num_heads, num_tokens, num_tokens]
        keys.transpose(2, 3); // restore keys

        for (int i = 0; i < batch; ++i) 
        {
            for (int h = 0; h < m_num_heads; ++h) 
            {
                for (int j = 0; j < num_tokens; ++j) 
                {
                    for (int k = 0; k < num_tokens; ++k) 
                    {
                        if (m_mask[j][k].value()) 
                        {
                            // Apply mask to attention scores
                            DType num = -std::numeric_limits<DType>::infinity();
                            attn_scores[i][h][j][k] = Tensor<DType>({}, &num);
                        } 
                    }
                }
            }
        }

        auto attn_weights = attn_scores / static_cast<DType>(std::sqrt(keys.dims().back()));
        attn_weights.softmax();

        auto context_vec = MatMul(attn_weights, values); // Shape: [batch, num_heads, num_tokens, head_dim]

        context_vec.transpose(1, 2); // Shape: [batch, num_tokens, num_heads, head_dim]

        context_vec.view({batch, num_tokens, m_d_out}); // Shape: [batch, num_tokens, d_out]
        return m_out_proj(context_vec); // Final projection to output dimension
    }

public:
    Dim m_d_out;
    Dim m_num_heads;
    Dim m_head_dim = m_d_out / m_num_heads;
    
    Linear<DType> m_W_query;
    Linear<DType> m_W_key;
    Linear<DType> m_W_value;
    Linear<DType> m_out_proj;
    Tensor<DType> m_mask; // Mask for attention, shape: [context_length_, context_length_]
};


template <typename DType>
class GELU
{
public:
    using Dim = typename Tensor<DType>::Dim;
    
    Tensor<DType> operator()(const Tensor<DType> &x) const
    {
        auto tmp = std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x);
        for (size_t i = 0; i < tmp.numel(); ++i) 
        {
            static_cast<DType*>(tmp.m_data)[i] = std::tanh(static_cast<DType*>(tmp.m_data)[i]);
        }
        return 0.5 * x * (1 + tmp);
    }
};


template <typename DType>
class FeedForward
{
public:
    using Dim = typename Tensor<DType>::Dim;

    FeedForward(Dim emb_dim)
        : m_linear1(emb_dim, 4 * emb_dim),
          m_gelu(),
          m_linear2(4 * emb_dim, emb_dim) {}

    Tensor<DType> operator()(const Tensor<DType> &x) const
    {
        // Apply first linear layer and GELU activation
        auto x1 = m_linear1(x);
        auto x2 = m_gelu(x1);
        // Apply second linear layer
        return m_linear2(x2);
    }

public:
    Linear<DType> m_linear1; // First linear layer
    GELU<DType> m_gelu; // GELU activation
    Linear<DType> m_linear2; // Second linear layer
};


template <typename DType>
class LayerNorm
{
public:
    using Dim = typename Tensor<DType>::Dim;

    LayerNorm(Dim emb_dim)
        : m_scale({emb_dim}),
          m_shift({emb_dim}) {}

    Tensor<DType> operator()(const Tensor<DType> &x) const
    {
        // Compute mean and variance
        auto mean = x, var = x;
        mean.mean(true);
        var.var(true, false);
        
        auto tmp = var + m_eps;
        for (size_t i = 0; i < tmp.numel(); ++i) 
        {
            static_cast<DType*>(tmp.m_data)[i] = std::sqrt(static_cast<DType*>(tmp.m_data)[i]);
        }
        auto norm_x = (x - mean) / tmp;
        return norm_x * m_scale + m_shift; // Apply scale and shift
    }

public:
    Float64 m_eps = 1e-5;
    Tensor<DType> m_scale; // Scale parameter
    Tensor<DType> m_shift; // Shift parameter
};


template <typename DType>
class TransformerBlock
{
public:
    using Dim = typename Tensor<Float32>::Dim;

    TransformerBlock(Dim d_in_, Dim d_out_, int context_length_, int num_heads_, bool qkv_bias_ = false)
        : m_att(d_in_, d_out_, context_length_, num_heads_, qkv_bias_),
          m_ff(d_out_),
          m_norm1(d_out_),
          m_norm2(d_out_)
    {
    }

    Tensor<DType> operator()(const Tensor<DType> &x) const
    {
        auto short_cut = x;
        auto x1 = m_norm1(x);
        x1 = m_att(x1);
        x1 = x1 + short_cut;

        short_cut = x1;
        auto x2 = m_norm2(x1);
        x2 = m_ff(x2);
        x2 = x2 + short_cut;

        return x2;
    }

public:
    MultiHeadAttention<DType> m_att;
    FeedForward<DType> m_ff;
    LayerNorm<DType> m_norm1;
    LayerNorm<DType> m_norm2;
};


template <typename DType>
class GPTModel
{
public:
    using Dim = typename Tensor<DType>::Dim;

    GPTModel(Dim vocab_size_, Dim emb_dim_, int context_length_, int num_heads_, int num_layers_, bool qkv_bias_ = false)
        : m_tok_embedding(vocab_size_, emb_dim_),
          m_pos_embedding(context_length_, emb_dim_),
          m_trf_blocks(num_layers_, TransformerBlock<DType>(emb_dim_, emb_dim_, context_length_, num_heads_, qkv_bias_)),
          m_final_norm(emb_dim_),
          m_out_head(emb_dim_, vocab_size_)
    {
    }

    Tensor<DType> operator()(const Tensor<TokenIndex> &in_idx) const
    {
        // auto batch_size = in_idx.dims()[0];
        auto seq_len = in_idx.dims()[1];
        auto tok_embeds = m_tok_embedding(in_idx);
        PosIndex *pos_indices = new PosIndex[seq_len];
        for (int i = 0; i < seq_len; ++i) 
            pos_indices[i] = static_cast<PosIndex>(i);
        auto pos_embeds = m_pos_embedding(Tensor<PosIndex>({seq_len}, pos_indices, true));
        auto x = tok_embeds + pos_embeds; // Add token and position embeddings
        for (const auto &block : m_trf_blocks)
        {
            x = block(x); // Pass through each transformer block
        }
        x = m_final_norm(x); // Final layer normalization
        auto logits = m_out_head(x);
        return logits; // Return the final logits
    }

public:
    Embedding<DType> m_tok_embedding;
    Embedding<DType> m_pos_embedding;
    std::vector<TransformerBlock<DType>> m_trf_blocks; // List of transformer blocks
    LayerNorm<DType> m_final_norm; // Final layer normalization
    Linear<DType> m_out_head;
};


struct TensorUnion
{
    DataType dtype;
    void* tensor;
};

std::unordered_map<std::string, TensorUnion> load_parameters(const std::string& filename, bool verbose = false) 
{
    std::unordered_map<std::string, TensorUnion> params;
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file) 
    {
        std::cerr << "Error opening file." << std::endl;
        return params;
    }
    while (true) 
    {
        ParamKeyLen key_len = 0;
        file.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
        if (file.eof()) {
            file.close();
            break; // End of file reached
        }
        std::string key(key_len, '\0');
        file.read(&key[0], key_len);
        if (verbose)
            std::cout << "Key: " << key << std::endl;
        ParamDType dtype = 0;
        file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
        ParamShapeLen shape_len = 0;
        file.read(reinterpret_cast<char*>(&shape_len), sizeof(shape_len));
        std::vector<ParamShapeElement> shape;
        ParamShapeElement data_size = 1;
        for (ParamShapeLen i = 0; i < shape_len; ++i) 
        {
            ParamShapeElement elem = 0;
            file.read(reinterpret_cast<char*>(&elem), sizeof(elem));
            shape.emplace_back(elem);
            data_size *= elem;
        }
        auto print_tensor = [](const std::string& key, const auto& tensor) {
            std::cout << "Loaded parameter: " << key << " with shape (";
            for (const auto& dim : tensor.dims())
                std::cout << dim << ",";
            std::cout << ")" << std::endl;
        };
        TensorUnion tensor;
        if (dtype == 1) // FLOAT_32
        { 
            Float32* data = new Float32[data_size];
            file.read(reinterpret_cast<char*>(data), data_size * sizeof(Float32));
            tensor.dtype = DataType::FLOAT_32;
            tensor.tensor = new Tensor(shape, data, true);
            if (verbose)
                print_tensor(key, *static_cast<Tensor<Float32>*>(tensor.tensor));
        } 
        else if (dtype == 2) // FLOAT_64
        {
            Float64* data = new Float64[data_size];
            file.read(reinterpret_cast<char*>(data), data_size * sizeof(Float64));
            tensor.dtype = DataType::FLOAT_64;
            tensor.tensor = new Tensor(shape, data, true);
            if (verbose)
                print_tensor(key, *static_cast<Tensor<Float64>*>(tensor.tensor));
        } 
        else 
            throw std::runtime_error("Unsupported dtype");
        
        params[key] = tensor;
    }
    file.close();
    return params;
}