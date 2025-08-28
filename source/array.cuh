#ifndef STN_CUDA_ARRAY_CUH
#define STN_CUDA_ARRAY_CUH

template<typename Item, unsigned int _n>
struct Array {
    Item item;
    Array<Item, _n - 1> tail;

    static __device__ __host__
    unsigned int n() {
        return _n;
    }

    template<unsigned int _i>
    __device__ __host__
    Item get() const {
        if constexpr (_i == 0)
            return item;
        else
            return tail.template get<_i - 1>();
    }

    template<typename... Items>
    static __device__ __host__
    Array of(Item _item, Items... _items) {
        return {_item, Array<Item, _n - 1>::of(_items...)};
    }
};

template<typename Item>
struct Array<Item, 0u> {
    static __device__ __host__
    unsigned int n() {
        return 0;
    }

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
