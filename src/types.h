#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <numeric>
#include <functional>

#include <cstdint>
#include <cstring>
#include <cmath>
#include <cassert>


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


/********** class TensorBase **********/
template <typename DType> class Tensor;  // forward declaration
template <typename DType> class TensorMap;  // forward declaration

template <typename DType>
class TensorBase
{
public:
    template <typename T> friend std::ostream& operator<<(std::ostream& os, const TensorBase<T> &tensor);
    using Dim = size_t;
    virtual ~TensorBase() = 0;

private:
    std::string tensor_data_string() const;

public:
    DataType dtype() const { return dtype_; }
    const std::vector<Dim> &dims() const { return dims_; }
    Dim dims(long long i) const { return dims_[i < 0 ? i + dims_.size() : i]; }
    const DType* data() const { return data_; }
    DType* data() { return data_; }
    size_t numel() const;

    void broadcast(Dim num);
    TensorMap<DType> map(Dim index);
    const TensorMap<DType> map(Dim index) const;
    DType& value(Dim index);
    const DType& value(Dim index) const;

    // Swap two dimensions
    Tensor<DType> transpose(Dim d1, Dim d2) const;
    // Softmax operation along the last dimension
    Tensor<DType> softmax() const;
    // Current only support last dimension
    Tensor<DType> mean(bool keep_dim = false) const;
    // Current only support last dimension
    Tensor<DType> var(bool keep_dim = false, bool unbiased = true) const;

public:
    constexpr static DataType dtype_ = dtype_enum_value<DType>::v;
    std::vector<Dim> dims_;
    DType *data_ = nullptr;
};

template <typename DType>
TensorBase<DType>::~TensorBase() {}

template <typename DType>
std::string TensorBase<DType>::tensor_data_string() const
{
    auto &tensor = *this;
    std::string str;
    if (tensor.dims_.size() == 1) 
    {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(4);
        oss << "[";
        if (tensor.dims_[0] < 10) 
        {
            for (Dim i = 0; i < tensor.dims_[0]; ++i)
            {
                oss << tensor.value(i);
                oss << ", ";
            }
        }
        else  // if the first dimension is large, we only show the first 3 elements and the last 3 elements
        {
            for (Dim i = 0; i < 3; ++i)
            {
                oss << tensor.value(i);
                oss << ", ";
            }
            oss << "..., ";
            for (Dim i = tensor.dims_[0] - 3; i < tensor.dims_[0]; ++i)
            {
                oss << tensor.value(i);
                oss << ", ";
            }
        }
        str += oss.str();
        // remove last comma and space and add closing bracket
        str.pop_back();
        str.back() = ']';
    } 
    else 
    {
        str += "[";
        if (tensor.dims_[0] < 10) 
        {
            for (Dim i = 0; i < tensor.dims_[0]; ++i)
                str += tensor.map(i).tensor_data_string() + "," + std::string(tensor.dims_.size() - 1, '\n');
        } 
        else  // if the first dimension is large, we only show the first 3 elements and the last 3 elements
        {
            for (Dim i = 0; i < 3; ++i)
                str += tensor.map(i).tensor_data_string() + "," + std::string(tensor.dims_.size() - 1, '\n');
            str += "...,\n";
            for (Dim i = tensor.dims_[0] - 3; i < tensor.dims_[0]; ++i)
                str += tensor.map(i).tensor_data_string() + "," + std::string(tensor.dims_.size() - 1, '\n');
        }
        // remove last ",\n\n..." and add closing bracket
        for (size_t i = 0; i < tensor.dims_.size() - 1; ++i)
            str.pop_back();
        str.back() = ']';
    }
    return str;
}

template <typename DType>
size_t TensorBase<DType>::numel() const 
{
    size_t num = 1;
    for (const auto &dim : this->dims_)
        num *= dim;
    return num;
}

template <typename DType>
void TensorBase<DType>::broadcast(Dim num)
{ 
    std::vector<Dim> add_dims(num, 1);
    dims_.insert(dims_.begin(), add_dims.begin(), add_dims.end());
}

template <typename DType>
TensorMap<DType> TensorBase<DType>::map(Dim index)
{
    if (this->dims_.size() < 2)
        throw std::out_of_range("Tensor with less than 2 dimensions cannot be mapped to TensorMap");
    return TensorMap<DType>(std::vector<Dim>(this->dims_.begin() + 1, this->dims_.end()), this->data_ + index * this->numel() / this->dims_[0]);
}

template <typename DType>
const TensorMap<DType> TensorBase<DType>::map(Dim index) const
{
    if (this->dims_.size() < 2)
        throw std::out_of_range("Tensor with less than 2 dimensions cannot be mapped to TensorMap");
    return TensorMap<DType>(std::vector<Dim>(this->dims_.begin() + 1, this->dims_.end()), this->data_ + index * this->numel() / this->dims_[0]);
}

template <typename DType>
DType& TensorBase<DType>::value(Dim index)
{
    if (this->dims_.size() != 1)
        throw std::out_of_range("Only 1-D tensor can access value by index");
    if (index >= this->dims_[0])
        throw std::out_of_range("Index out of range");
    return this->data_[index];
}

template <typename DType>
const DType& TensorBase<DType>::value(Dim index) const
{
    if (this->dims_.size() != 1)
        throw std::out_of_range("Only 1-D tensor can access value by index");
    if (index >= this->dims_[0])
        throw std::out_of_range("Index out of range");
    return this->data_[index];
}

template <typename DType>
Tensor<DType> TensorBase<DType>::transpose(Dim d1, Dim d2) const
{
    if (d1 < 0 || d1 >= static_cast<Dim>(dims_.size()) || d2 < 0 || d2 >= static_cast<Dim>(dims_.size()))
        throw std::out_of_range("Dimension index out of range");

    if (d1 == d2)
        return Tensor<DType>(*this);  // No need to transpose if the dimensions are the same

    // prepare two strides vectors to calculate the index
    auto swap_dims = dims_;
    std::swap(swap_dims[d1], swap_dims[d2]);
    std::vector<Dim> strides{1}, swap_strides{1};
    for (auto it = dims_.rbegin(); it != dims_.rend() - 1; ++it)
        strides.push_back(*it * strides.back());
    for (auto it = swap_dims.rbegin(); it != swap_dims.rend() - 1; ++it)
        swap_strides.push_back(*it * swap_strides.back());
    std::reverse(strides.begin(), strides.end());
    std::reverse(swap_strides.begin(), swap_strides.end());

    Tensor<DType> result(swap_dims);

    std::function<void (std::vector<Dim> &index)> transpose_recurse = [this, d1, d2, &strides, &swap_strides, &result, &transpose_recurse](std::vector<Dim> &index) {
        if (index.size() == this->dims_.size()) 
        {
            auto swap_index = index;
            std::swap(swap_index[d1], swap_index[d2]);
            size_t data_index = 0, swap_data_index = 0;
            for (size_t i = 0; i < index.size(); ++i) 
            {
                data_index += index[i] * strides[i];
                swap_data_index += swap_index[i] * swap_strides[i];
            }
            result.data_[swap_data_index] = static_cast<DType*>(data_)[data_index];
        }
        else
        {
            for (Dim i = 0; i < this->dims_[index.size()]; ++i) 
            {
                index.push_back(i);
                transpose_recurse(index);
                index.pop_back();
            }
        }
    };

    std::vector<Dim> index;
    transpose_recurse(index);
    return result;
}

template <typename DType>
Tensor<DType> TensorBase<DType>::softmax() const
{
    Tensor<DType> result(*this);
    Dim offset = 0;
    while (offset < result.numel())
    {
        TensorMap<DType> slice({result.dims(-1)}, result.data() + offset);
        DType max_val = *std::max_element(slice.data(), slice.data() + slice.dims(-1));
        std::for_each(slice.data(), slice.data() + slice.dims(-1), [&](DType &val) {
            if (std::isnan(val) || std::isinf(val)) 
                val = 0; // Handle NaN and Inf values: -std::numeric_limits<DType>::infinity()
            else
                val = std::exp(val - max_val);
        });
        DType sum = std::accumulate(slice.data(), slice.data() + slice.dims(-1), static_cast<DType>(0));
        if (std::isnan(sum) || std::isinf(sum) || sum == 0)
            throw std::runtime_error("Softmax sum is NaN, Inf or zero");
        std::for_each(slice.data(), slice.data() + slice.dims(-1), [sum](DType &val) {
            val /= sum;
        });
        offset += result.dims(-1);
    }
    return result;
}

template <typename DType>
Tensor<DType> TensorBase<DType>::mean(bool keep_dim) const
{
    auto result_dims = dims();
    result_dims.back() = 1;
    Tensor<DType> result(result_dims);

    Dim offset = 0;
    while (offset < numel())
    {
        TensorMap<DType> slice(std::vector<Dim>{dims_.back()}, data_ + offset);
        DType sum = std::accumulate(slice.data(), slice.data() + dims_.back(), static_cast<DType>(0));
        if (std::isnan(sum) || std::isinf(sum) || sum == 0)
            throw std::runtime_error("Mean sum is NaN, Inf or zero");
        DType mean_val = sum / dims_.back();
        result.data_[offset / dims_.back()] = mean_val;
        offset += dims_.back();
    }

    if (result.dims_.size() == 1)
        keep_dim = true;
    if (!keep_dim)
        result.view(std::vector<Dim>(result.dims_.begin(), result.dims_.end() - 1));
    return result;
}

template <typename DType>
Tensor<DType> TensorBase<DType>::var(bool keep_dim, bool unbiased) const
{
    auto result_dims = dims();
    result_dims.back() = 1;
    Tensor<DType> result(result_dims);

    Dim offset = 0;
    while (offset < numel())
    {
        TensorMap<DType> slice(std::vector<Dim>{dims_.back()}, data_ + offset);
        DType sum = std::accumulate(slice.data(), slice.data() + dims_.back(), static_cast<DType>(0));
        if (std::isnan(sum) || std::isinf(sum) || sum == 0)
            throw std::runtime_error("Variance mean sum is NaN, Inf or zero");
        DType mean_val = sum / dims_.back();
        DType var_sum = 0;
        for (Dim i = 0; i < slice.dims().back(); ++i) 
        {
            var_sum += (slice.data()[i] - mean_val) * (slice.data()[i] - mean_val);
        }
        if (unbiased && dims_.back() > 1) 
        {
            result.data_[offset / dims_.back()] = var_sum / (dims_.back() - 1);
        } 
        else 
        {
            result.data_[offset / dims_.back()] = var_sum / dims_.back();
        }
        offset += dims_.back();
    }

    if (result.dims_.size() == 1)
        keep_dim = true;
    if (!keep_dim)
        result.view(std::vector<Dim>(result.dims_.begin(), result.dims_.end() - 1));
    return result;
}


/********** class Tensor **********/
template <typename DType>
class Tensor : public TensorBase<DType>
{
public:
    using Dim = typename TensorBase<DType>::Dim;

    // Tensor has 1 dimension with 1 element at least
    Tensor();

    // Support 4 dimension at most by initializer. 
    template <typename U, typename = std::enable_if_t<std::is_fundamental_v<U> && std::is_convertible_v<U, DType>>>
    Tensor(const std::initializer_list<U> &data);  
    template <typename U, typename = std::enable_if_t<std::is_fundamental_v<U> && std::is_convertible_v<U, DType>>>
    Tensor(const std::initializer_list<std::initializer_list<U>> &data);
    template <typename U, typename = std::enable_if_t<std::is_fundamental_v<U> && std::is_convertible_v<U, DType>>>
    Tensor(const std::initializer_list<std::initializer_list<std::initializer_list<U>>> &data);
    template <typename U, typename = std::enable_if_t<std::is_fundamental_v<U> && std::is_convertible_v<U, DType>>>
    Tensor(const std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>> &data);

    template <typename... Args, typename = std::enable_if_t<(std::is_convertible_v<std::decay_t<Args>, Dim> && ...) && (sizeof...(Args) > 0)>>
    Tensor(Args&&... args);

    Tensor(std::vector<Dim> dims);
    Tensor(std::vector<Dim> dims, const DType* data);
    Tensor(const Tensor &other);
    Tensor& operator=(const Tensor &other);
    Tensor(Tensor &&other) noexcept;
    Tensor& operator=(Tensor &&other) noexcept;
    ~Tensor(); 

    Tensor(const TensorBase<DType> &tensorbase) : TensorBase<DType>()
    {
        this->dims_ = tensorbase.dims_;
        this->data_ = new DType[this->numel()]{};
        std::memcpy(this->data_, tensorbase.data_, this->numel() * sizeof(DType));
    }

    Tensor& resize(std::vector<Dim> new_dims);
    Tensor& view(std::vector<Dim> new_dims);
};

template <typename DType>
Tensor<DType>::Tensor() : TensorBase<DType>() 
{
    this->dims_ = {1};
    this->data_ = new DType[1]{};
}

template <typename DType>
 template <typename U, typename>
Tensor<DType>::Tensor(const std::initializer_list<U> &data)
    : TensorBase<DType>()
{
    this->dims_ = {data.size()};
    this->data_ = new DType[this->numel()]{};
    std::copy(data.begin(), data.end(), this->data_);
}

template <typename DType>
 template <typename U, typename>
Tensor<DType>::Tensor(const std::initializer_list<std::initializer_list<U>> &data)
    : TensorBase<DType>()
{
    this->dims_ = {data.size(), data.begin()->size()};
    this->data_ = new DType[this->numel()]{};

    size_t offset = 0;
    for (const auto& row : data) 
    {
        if (row.size() != this->dims_[1]) 
            throw std::runtime_error("Inconsistent size in dimension 1");
        std::copy(row.begin(), row.end(), this->data_ + offset);
        offset += row.size();
    }
}

template <typename DType>
 template <typename U, typename>
Tensor<DType>::Tensor(const std::initializer_list<std::initializer_list<std::initializer_list<U>>> &data)
    : TensorBase<DType>()
{
    this->dims_ = {data.size(), data.begin()->size(), data.begin()->begin()->size()};
    this->data_ = new DType[this->numel()]{};

    size_t offset = 0;
    for (const auto& matrix : data) 
    {
        if (matrix.size() != this->dims_[1]) 
            throw std::runtime_error("Inconsistent size in dimension 1");
        for (const auto& row : matrix) 
        {
            if (row.size() != this->dims_[2]) 
                throw std::runtime_error("Inconsistent size in dimension 2");
            std::copy(row.begin(), row.end(), this->data_ + offset);
            offset += row.size();
        }
    }
}

template <typename DType>
 template <typename U, typename>
Tensor<DType>::Tensor(const std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>> &data)
    : TensorBase<DType>()
{
    this->dims_ = {data.size(), data.begin()->size(), data.begin()->begin()->size(), data.begin()->begin()->begin()->size()};
    this->data_ = new DType[this->numel()]{};

    size_t offset = 0;
    for (const auto& tensor : data) 
    {
        if (tensor.size() != this->dims_[1]) 
            throw std::runtime_error("Inconsistent size in dimension 1");
        for (const auto& matrix : tensor) 
        {
            if (matrix.size() != this->dims_[2]) 
                throw std::runtime_error("Inconsistent size in dimension 2");
            for (const auto& row : matrix) 
            {
                if (row.size() != this->dims_[3]) 
                    throw std::runtime_error("Inconsistent size in dimension 3");
                std::copy(row.begin(), row.end(), this->data_ + offset);
                offset += row.size();
            }
        }
    }
}

template <typename DType>
 template<typename... Args, typename>
Tensor<DType>::Tensor(Args&&... args)
    : TensorBase<DType>()
{
    this->dims_ = {static_cast<Dim>(args)...};
    this->data_ = new DType[this->numel()]{};

    // Check dimensions
    std::for_each(this->dims_.begin(), this->dims_.end(), [](Dim dim) { if (dim <= 0) throw std::runtime_error("Dimension element must be greater than 0"); });
}

template <typename DType>
Tensor<DType>::Tensor(std::vector<Dim> dims)
    : TensorBase<DType>()
{
    this->dims_ = std::move(dims);
    this->data_ = new DType[this->numel()]{};

    // Check dimensions
    if (this->dims_.empty()) 
        throw std::runtime_error("Dimension size must be greater than 0");
    std::for_each(this->dims_.begin(), this->dims_.end(), [](Dim dim) { if (dim <= 0) throw std::runtime_error("Dimension element must be greater than 0"); });
}

template <typename DType>
Tensor<DType>::Tensor(std::vector<Dim> dims, const DType* data)
    : TensorBase<DType>()
{
    this->dims_ = std::move(dims);
    this->data_ = new DType[this->numel()]{};

    // Check dimensions
    if (this->dims_.empty()) 
        throw std::runtime_error("Dimension size must be greater than 0");
    std::for_each(this->dims_.begin(), this->dims_.end(), [](Dim dim) { if (dim <= 0) throw std::runtime_error("Dimension element must be greater than 0"); });

    std::memcpy(this->data_, data, this->numel() * sizeof(DType));
}

template <typename DType>
Tensor<DType>::Tensor(const Tensor &other)
    : TensorBase<DType>(other)
{
    this->dims_ = other.dims_;
    this->data_ = new DType[this->numel()]{};
    std::memcpy(this->data_, other.data_, this->numel() * sizeof(DType));
}

template <typename DType>
Tensor<DType>& Tensor<DType>::operator=(const Tensor &other) 
{
    if (this != &other) 
    {
        if (this->numel() == other.numel()) 
        {
            // Same size, no need to reallocate
            std::memcpy(this->data_, other.data_, this->numel() * sizeof(DType));
            return *this;
        }

        delete[] static_cast<DType*>(this->data_);
        this->dims_ = other.dims_;
        this->data_ = new DType[this->numel()]{};
        std::memcpy(this->data_, other.data_, this->numel() * sizeof(DType));
    }
    return *this;
}

template <typename DType>
Tensor<DType>::Tensor(Tensor &&other) noexcept
    : TensorBase<DType>(std::move(other))
{
    this->dims_ = std::move(other.dims_);
    this->data_ = other.data_;
    other.data_ = nullptr;  // Transfer ownership 
}

template <typename DType>
Tensor<DType>& Tensor<DType>::operator=(Tensor &&other) noexcept
{
    if (this != &other) 
    {
        delete[] static_cast<DType*>(this->data_);
        this->dims_ = std::move(other.dims_);
        this->data_ = other.data_;
        other.data_ = nullptr; // Transfer ownership
    }
    return *this;
}

template <typename DType>
Tensor<DType>::~Tensor() 
{
    delete[] static_cast<DType*>(this->data_);
    this->data_ = nullptr;
}

template <typename DType>
Tensor<DType>& Tensor<DType>::resize(std::vector<Dim> new_dims)
{
    Tensor<DType> temp(new_dims);
    std::memcpy(temp.data_, this->data_, std::min(temp.numel(), this->numel()) * sizeof(DType));
    *this = std::move(temp);
    return *this;
}

template <typename DType>
Tensor<DType>& Tensor<DType>::view(std::vector<Dim> new_dims) 
{
    this->dims_ = std::move(new_dims);
    return *this;
}


/********** class TensorMap **********/
template <typename DType>
class TensorMap : public TensorBase<DType>
{
public:
    using Dim = typename TensorBase<DType>::Dim;

    TensorMap(const TensorBase<DType> &tensor);
    TensorMap(const Tensor<DType> &tensor);
    TensorMap(std::vector<Dim> dims, DType *data);

    TensorMap(const TensorMap &other) = default;
    TensorMap& operator=(const TensorMap &other) = default;
    TensorMap(TensorMap &&other) noexcept = default;
    TensorMap& operator=(TensorMap &&other) noexcept = default;
    ~TensorMap() = default; 

    operator const Tensor<DType>() const { return Tensor<DType>(this->dims_, this->data_); }
    operator Tensor<DType>() { return Tensor<DType>(this->dims_, this->data_); }
};

template <typename DType>
TensorMap<DType>::TensorMap(const TensorBase<DType> &tensor) : TensorBase<DType>()
{
    this->dims_ = tensor.dims_;
    this->data_ = tensor.data_;
}

template <typename DType>
TensorMap<DType>::TensorMap(const Tensor<DType> &tensor) : TensorBase<DType>()
{
    this->dims_ = tensor.dims_;
    this->data_ = tensor.data_;
}

template <typename DType>
TensorMap<DType>::TensorMap(std::vector<Dim> dims, DType *data) : TensorBase<DType>()
{
    this->dims_ = dims;
    this->data_ = data;

    // Check dimensions
    if (this->dims_.empty()) 
        throw std::runtime_error("Dimension size must be greater than 0");
    std::for_each(this->dims_.begin(), this->dims_.end(), [](Dim dim) { if (dim <= 0) throw std::runtime_error("Dimension element must be greater than 0"); });
}


/********** Global functions **********/
template <typename DType>
std::ostream& operator<<(std::ostream& os, const TensorBase<DType> &tensor) 
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
        case DataType::INT_32:
            os << "Int32";
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

namespace
{
    template <typename DType>
    DType LinearMatMul1D(const TensorBase<DType> &tensor, const TensorBase<DType> &other)
    {
        assert(tensor.dims().size() == 1 && other.dims().size() == 1);
        assert(tensor.dims(0) == other.dims(0));
        DType result = 0;
        for (typename TensorBase<DType>::Dim i = 0; i < tensor.dims()[0]; ++i) 
        {
            result += tensor.data()[i] * other.data()[i];
        }
        return result;
    }
}

template <typename DType>
Tensor<DType> LinearMatMul(const TensorBase<DType> &tensor, const TensorBase<DType> &other)
{
    // broadcast the tensors
    TensorMap<DType> tm1 = tensor, tm2 = other;
    if (tm1.dims().size() > tm2.dims().size()) 
    {
        tm2.broadcast(tm1.dims().size() - tm2.dims().size());
    }
    else if (tm1.dims().size() < tm2.dims().size()) 
    {
        tm1.broadcast(tm2.dims().size() - tm1.dims().size());
    }

    std::vector<typename TensorMap<DType>::Dim> result_dims;
    for (size_t i = 0; i < tm1.dims().size() - 2; ++i) 
    {
        if (tm1.dims()[i] != tm2.dims()[i]) 
        {
            if (tm1.dims()[i] != 1 && tm2.dims()[i] != 1)
                throw std::runtime_error("The dimensions of the tensors must match or be 1 for broadcasting");
        }
        result_dims.push_back(std::max(tm1.dims()[i], tm2.dims()[i]));
    }
    result_dims.push_back(tm1.dims(-2));
    result_dims.push_back(tm2.dims(-2));

    if (tm1.dims(-1) != tm2.dims(-1))
        throw std::runtime_error("The last dimension of the two tensors must match for linear matrix multiplication");

    std::function<void(const TensorMap<DType>&, const TensorMap<DType>&, TensorMap<DType>&)> matmul_recurse = [&matmul_recurse](const TensorMap<DType> &a, const TensorMap<DType> &b, TensorMap<DType> &r) {
        if (a.dims().size() == 2) 
        {
            for (typename Tensor<DType>::Dim i = 0; i < r.dims()[0]; ++i) 
            {
                TensorMap<DType> row = a.map(i);
                for (typename Tensor<DType>::Dim j = 0; j < r.dims()[1]; ++j) 
                {
                    TensorMap<DType> col = b.map(j);
                    r.data()[i * r.dims()[1] + j] = LinearMatMul1D(row, col);
                }
            }
        } 
        else 
        {
            for (typename Tensor<DType>::Dim i = 0; i < r.dims()[0]; ++i) 
            {
                typename Tensor<DType>::Dim a_index = (a.dims()[0] == 1) ? 0 : i;
                typename Tensor<DType>::Dim b_index = (b.dims()[0] == 1) ? 0 : i;
                auto am = a.map(a_index);
                auto bm = b.map(b_index);
                auto rm = r.map(i);
                matmul_recurse(am, bm, rm);
            }
        }
    };

    Tensor<DType> result(result_dims);
    TensorMap<DType> result_map = result;
    matmul_recurse(tm1, tm2, result_map);
    return result;
}

template <typename DType>
Tensor<DType> MatMul(const TensorBase<DType> &tensor, const TensorBase<DType> &other) 
{
    if (tensor.dims().size() == 1 || other.dims().size() == 1) 
    {
        if (tensor.dims().size() == 1 && other.dims().size() == 1) 
        {
            if (tensor.numel() != other.numel())
                throw std::runtime_error("1D tensors must have the same number of elements for dot product");
            return Tensor<DType>{LinearMatMul1D(tensor, other)};
        }
        else
            throw std::runtime_error("The dimensions of the tensors must match or be 1 for broadcasting");
    }

    if (tensor.dims(-1) != other.dims(-2))
        throw std::runtime_error("The last dimension of the first tensor must match the second to last dimension of the second tensor");

    auto other_T = other.transpose(other.dims().size() - 2, other.dims().size() - 1);  // Transpose the second tensor for easier multiplication
    return LinearMatMul(tensor, other_T);
}

namespace
{
    template <typename DType, typename Operator>
    Tensor<DType> BasicTensorOperator(const TensorBase<DType> &tensor, const TensorBase<DType> &other, const Operator &op) 
    {
        // broadcast the tensors
        TensorMap<DType> tm1 = tensor , tm2 = other;
        if (tm1.dims().size() > tm2.dims().size())
        {
            tm2.broadcast(tm1.dims().size() - tm2.dims().size());
        }
        else if (tm1.dims().size() < tm2.dims().size()) 
        {
            tm1.broadcast(tm2.dims().size() - tm1.dims().size());
        }
        
        std::vector<typename TensorMap<DType>::Dim> result_dims;
        for (size_t i = 0; i < tm1.dims().size(); ++i) 
        {
            if (tm1.dims()[i] != tm2.dims()[i]) 
            {
                if (tm1.dims()[i] != 1 && tm2.dims()[i] != 1)
                    throw std::runtime_error("The dimensions of the tensors must match or be 1 for broadcasting");
            }
            result_dims.push_back(std::max(tm1.dims()[i], tm2.dims()[i]));
        }

        std::function<void(const TensorMap<DType>&, const TensorMap<DType>&, TensorMap<DType>&)> operator_recurse = [&op, &operator_recurse](const TensorMap<DType> &a, const TensorMap<DType> &b, TensorMap<DType> &r) {
            if (a.dims().size() == 1)
            {
                for (typename TensorMap<DType>::Dim i = 0; i < r.dims(0); ++i) 
                {
                    typename TensorMap<DType>::Dim a_index = (a.dims(0) == 1) ? 0 : i;
                    typename TensorMap<DType>::Dim b_index = (b.dims(0) == 1) ? 0 : i;
                    r.value(i) = op(a.value(a_index), b.value(b_index));
                }
            }
            else
            {
                for (typename TensorMap<DType>::Dim i = 0; i < r.dims(0); ++i) 
                {
                    typename TensorMap<DType>::Dim a_index = i, b_index = i;
                    if (a.dims(0) == 1)
                        a_index = 0;
                    if (b.dims(0) == 1)
                        b_index = 0;
                    auto am = a.map(a_index);
                    auto bm = b.map(b_index);
                    auto rm = r.map(i);
                    operator_recurse(am, bm, rm);
                }
            }
        };

        Tensor<DType> result(result_dims);
        TensorMap<DType> result_map = result;
        operator_recurse(tm1, tm2, result_map);
        return result;
    }
}

template <typename DType>
Tensor<DType> operator+(const TensorBase<DType> &tensor, const TensorBase<DType> &other) 
{
    return BasicTensorOperator(tensor, other, std::plus<DType>());
}

template <typename DType>
Tensor<DType> operator-(const TensorBase<DType> &tensor, const TensorBase<DType> &other) 
{
    return BasicTensorOperator(tensor, other, std::minus<DType>());
}

template <typename DType>
Tensor<DType> operator*(const TensorBase<DType> &tensor, const TensorBase<DType> &other) 
{
    return BasicTensorOperator(tensor, other, std::multiplies<DType>());
}

template <typename DType>
Tensor<DType> operator/(const TensorBase<DType> &tensor, const TensorBase<DType> &other) 
{
    // undefined behavior for std::divides if dividing by zero, use own implementation
    return BasicTensorOperator(tensor, other, [](DType a, DType b) { 
        if (b == 0) 
            throw std::runtime_error("Division by zero");
        return a / b; 
    });
}

template <typename DType, typename ScalarType, 
          typename = std::enable_if_t<std::is_convertible_v<ScalarType, DType>>>
Tensor<DType> operator+(const TensorBase<DType> &tensor, ScalarType scalar)
{

    Tensor<DType> result(tensor);
    for (size_t i = 0; i < result.numel(); ++i) 
    {
        static_cast<DType*>(result.data())[i] += static_cast<DType>(scalar);
    }
    return result;
}

template <typename DType, typename ScalarType, 
          typename = std::enable_if_t<std::is_convertible_v<ScalarType, DType>>>
Tensor<DType> operator+(ScalarType scalar, const TensorBase<DType> &tensor)
{
    return tensor + scalar; // Use the existing operator+ for consistency
}

template <typename DType, typename ScalarType,
          typename = std::enable_if_t<std::is_convertible_v<ScalarType, DType>>>
Tensor<DType> operator-(const TensorBase<DType> &tensor, ScalarType scalar)
{
    Tensor<DType> result(tensor);
    for (size_t i = 0; i < result.numel(); ++i) 
    {
        static_cast<DType*>(result.data())[i] -= static_cast<DType>(scalar);
    }
    return result;
}

template <typename DType, typename ScalarType, 
          typename = std::enable_if_t<std::is_convertible_v<ScalarType, DType>>>
Tensor<DType> operator-(ScalarType scalar, const TensorBase<DType> &tensor)
{
    Tensor<DType> result(tensor);
    for (size_t i = 0; i < result.numel(); ++i) 
    {
        static_cast<DType*>(result.data())[i] = static_cast<DType>(scalar) - static_cast<DType*>(result.data())[i];
    }
    return result;
}

template <typename DType, typename ScalarType, 
          typename = std::enable_if_t<std::is_convertible_v<ScalarType, DType>>>
Tensor<DType> operator*(const TensorBase<DType> &tensor, ScalarType scalar) 
{
    Tensor<DType> result(tensor);
    for (size_t i = 0; i < result.numel(); ++i) 
    {
        static_cast<DType*>(result.data())[i] *= static_cast<DType>(scalar);
    }
    return result;
}

template <typename DType, typename ScalarType, 
          typename = std::enable_if_t<std::is_convertible_v<ScalarType, DType>>>
Tensor<DType> operator*(ScalarType scalar, const TensorBase<DType> &tensor)
{
    return tensor * scalar; // Use the existing operator* for consistency
}

template <typename DType, typename ScalarType, 
          typename = std::enable_if_t<std::is_convertible_v<ScalarType, DType>>>
Tensor<DType> operator/(const TensorBase<DType> &tensor, ScalarType scalar) 
{
    if (scalar == 0)
        throw std::runtime_error("Cannot divide by zero");

    Tensor<DType> result(tensor);
    for (size_t i = 0; i < result.numel(); ++i) 
    {
        static_cast<DType*>(result.data())[i] /= static_cast<DType>(scalar);
    }
    return result;
}

template <typename DType, typename ScalarType, 
          typename = std::enable_if_t<std::is_convertible_v<ScalarType, DType>>>
Tensor<DType> operator/(ScalarType scalar, const TensorBase<DType> &tensor)
{
    Tensor<DType> result(tensor);
    for (size_t i = 0; i < result.numel(); ++i) 
    {
        if (static_cast<DType*>(result.data())[i] == 0)
            throw std::runtime_error("Cannot divide by zero");
        static_cast<DType*>(result.data())[i] = static_cast<DType>(scalar) / static_cast<DType*>(result.data())[i];
    }
    return result;
}


/***** Model related types *****/
template <typename DType>
class Embedding
{
public:
    using Dim = typename Tensor<DType>::Dim;

    Embedding(Dim num_embeddings_, Dim embedding_dim_)
        : weight(num_embeddings_, embedding_dim_) {}
    
    template <typename Index, typename = std::enable_if_t<std::is_integral_v<Index>>>
    Tensor<DType> operator()(const Tensor<Index> &x) const
    {
        auto output_size = x.dims();
        output_size.push_back(weight.dims(-1));
        Tensor<DType> output(output_size);
        size_t chunk_bytes = output_size.back() * sizeof(DType);

        size_t offset = 0;
        while (offset < x.numel())
        {
            std::memcpy(output.data() + offset * output_size.back(),
                        weight.map((x.data() + offset)[0]).data(),
                        chunk_bytes);
            ++offset;
        }
        return output;
    }

public:
    Tensor<DType> weight;
};


template <typename DType>
class Linear 
{
public:
    using Dim = typename Tensor<DType>::Dim;

    Linear(Dim in_features_, Dim out_features_, bool bias_ = true)
        : weight(out_features_, in_features_), 
          bias_flag(bias_),
          bias(bias_ ? Tensor<DType>(out_features_) : Tensor<DType>()) {}

    Tensor<DType> operator()(const Tensor<DType> &x) const
    {
        Tensor<DType> output = LinearMatMul(x, weight);
        if (bias_flag) 
        {
            return output + bias;
        }
        return output;
    }

public:
    Tensor<DType> weight;
    bool bias_flag;
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
        m_mask = Tensor<DType>(static_cast<Dim>(context_length_), static_cast<Dim>(context_length_));
        DType *mask_data = m_mask.data();
        for (int i = 0; i < context_length_; ++i) 
        {
            for (int j = 0; j < context_length_; ++j) 
            {
                mask_data[i * context_length_ + j] = (i < j) ? 1 : 0;
            }
        }
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

        keys = keys.transpose(1, 2); // Shape: [batch, num_heads, num_tokens, head_dim]
        queries = queries.transpose(1, 2); // Shape: [batch, num_heads, num_tokens, head_dim]
        values = values.transpose(1, 2); // Shape: [batch, num_heads, num_tokens, head_dim]

        auto attn_scores = MatMul(queries, keys.transpose(2, 3)); // Shape: [batch, num_heads, num_tokens, num_tokens]

        for (Dim i = 0; i < batch; ++i) 
        {
            for (Dim h = 0; h < m_num_heads; ++h) 
            {
                for (Dim j = 0; j < num_tokens; ++j) 
                {
                    for (Dim k = 0; k < num_tokens; ++k) 
                    {
                        if (m_mask.map(j).value(k))
                        {
                            // Apply mask to attention scores
                            DType num = -std::numeric_limits<DType>::infinity();
                            attn_scores.map(i).map(h).map(j).value(k) = num;
                        } 
                    }
                }
            }
        }

        auto attn_weights = attn_scores / static_cast<DType>(std::sqrt(keys.dims().back()));
        attn_weights = attn_weights.softmax();

        auto context_vec = MatMul(attn_weights, values); // Shape: [batch, num_heads, num_tokens, head_dim]
        context_vec = context_vec.transpose(1, 2); // Shape: [batch, num_tokens, num_heads, head_dim]
        context_vec.view({batch, num_tokens, m_d_out}); // Shape: [batch, num_tokens, d_out]
        return m_out_proj(context_vec);
    }

public:
    Dim m_d_out;
    Dim m_num_heads;
    Dim m_head_dim = m_d_out / m_num_heads;
    
    Linear<DType> m_W_query;
    Linear<DType> m_W_key;
    Linear<DType> m_W_value;
    Linear<DType> m_out_proj;
    Tensor<DType> m_mask;
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
            static_cast<DType*>(tmp.data())[i] = std::tanh(static_cast<DType*>(tmp.data())[i]);
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
        auto x1 = m_linear1(x);
        auto x2 = m_gelu(x1);
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
        : m_scale(emb_dim),
          m_shift(emb_dim) {}

    Tensor<DType> operator()(const Tensor<DType> &x) const
    {
        // Compute mean and variance
        auto mean = x.mean(true), var = x.var(true, false) + m_eps;
        for (size_t i = 0; i < var.numel(); ++i) 
        {
            static_cast<DType*>(var.data())[i] = std::sqrt(static_cast<DType*>(var.data())[i]);
        }
        auto norm_x = (x - mean) / var;
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
          m_out_head(emb_dim_, vocab_size_, false)
    {
    }

    Tensor<DType> operator()(const Tensor<TokenIndex> &in_idx) const
    {
        // auto batch_size = in_idx.dims()[0];
        auto seq_len = in_idx.dims()[1];
        auto tok_embeds = m_tok_embedding(in_idx);
        std::vector<PosIndex> pos_indices(seq_len);
        for (Dim i = 0; i < seq_len; ++i) 
            pos_indices[i] = static_cast<PosIndex>(i);
        auto pos_embeds = m_pos_embedding(Tensor<PosIndex>({seq_len}, pos_indices.data()));
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


// Define types for reading parameters
using ParamKeyLen = int32_t;
using ParamDType = int32_t;
using ParamShapeLen = int32_t;
using ParamShapeElement = int32_t;

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
            std::vector<Float32> data(data_size);
            file.read(reinterpret_cast<char*>(data.data()), data_size * sizeof(Float32));
            tensor.dtype = DataType::FLOAT_32;
            tensor.tensor = new Tensor<Float32>(std::vector<size_t>(shape.begin(), shape.end()), data.data());
            if (verbose)
                print_tensor(key, *static_cast<Tensor<Float32>*>(tensor.tensor));
        } 
        else if (dtype == 2) // FLOAT_64
        {
            std::vector<Float64> data(data_size);
            file.read(reinterpret_cast<char*>(data.data()), data_size * sizeof(Float64));
            tensor.dtype = DataType::FLOAT_64;
            tensor.tensor = new Tensor<Float64>(std::vector<size_t>(shape.begin(), shape.end()), data.data());
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