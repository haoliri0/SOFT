#ifndef STN_CUDA_ARRAY_CUH
#define STN_CUDA_ARRAY_CUH

template<typename Item, unsigned int n>
struct Array {
    Item item;
    Array<Item, n - 1> tail;

    template<typename... Items>
    static __device__ __host__
    Array of(Item _item, Items... _items) {
        return {_item, Array<Item, n - 1>::of(_items...)};
    }

    template<unsigned int i>
    __device__ __host__
    Item &get() {
        if constexpr (i > 0)
            return tail.template get<i - 1>();
        else
            return item;
    }

    template<unsigned int i>
    __device__ __host__
    const Item &get() const {
        if constexpr (i > 0)
            return tail.template get<i - 1>();
        else
            return item;
    }

    __device__ __host__
    Item &get(const unsigned int i) {
        return reinterpret_cast<Item *>(this)[i];
    }

    __device__ __host__
    const Item &get(const unsigned int i) const {
        return reinterpret_cast<const Item *>(this)[i];
    }
};

template<typename Item>
struct Array<Item, 0u> {
    static __device__ __host__
    Array of() {
        return {};
    }
};

template<typename Item, typename... Items>
static __device__ __host__
Array<Item, sizeof...(Items)> arrayof(Items... items) {
    return Array<Item, sizeof...(Items)>::of(items...);
}

#endif
