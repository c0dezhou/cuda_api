#include <cuda.h>
#include <linux/kernel.h>
#include <experimental/tuple>
#include <iostream>
#include <tuple>

#if 0  // just verify looptest was enabled, skip at build

void checkError(int res) {
    if (res != 0) {
        std::cerr << "[PRINT] :" << res << ";\n" ;
    }
}

#define LOOP(func, times)                                       \
    do {                                                        \
        for (int i = 0; i < times; i++) {                       \
            std::cout << "loop calling " << #func << std::endl; \
            checkError(func);                                   \
            std::cout << "done calling " << #func << std::endl; \
        }                                                       \
    } while (0);

template <typename F, typename... Args>
class FuncPtr {
   public:
    FuncPtr(F (*func)(Args...)) : func_(func) {}

    F operator()(Args... args) { return func_(args...); }

   private:
    F (*func_)(Args...);
};

template <typename F, typename... Args>
FuncPtr<F, Args...> makeFuncPtr(F (*func)(Args...)) {
    return FuncPtr<F, Args...>(func);
};

template <typename F, typename... Args>
void loopFuncPtr(int times,
                 FuncPtr<F, Args...> funcPtr,
                 std::tuple<Args...> args) {
    LOOP(std::experimental::apply(funcPtr, args), times);
};

int test_do(int a){
    a++;
    std::cout << " surely do that " << a << std::endl;
    return 2;
}

int main() {
    auto testdo_func = makeFuncPtr(test_do);

    auto testdo_param = []() { return std::make_tuple(4); };

    loopFuncPtr(5, testdo_func, testdo_param());

    return 0;
}
#endif
