#include <experimental/tuple>
#include <tuple>
#include "test_utils.h"

#define LOOP(func, times)                 \
    do {                                  \
        for (int i = 0; i < times; i++) { \
            checkError(func);             \
        }                                 \
    } while (0);

#define PRINT_FUNCNAME(func)                                \
    do {                                                    \
        std::cout << "loop calling " << #func << std::endl; \
        func;                                               \
        std::cout << "done calling " << #func << std::endl; \
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