# cuda_driver_api

1. action_case: 
    1. AC_BA_funcname_describe  ----  base_action case
    2. AC_EG_funcname_describe  ----  edge_test case
    3. AC_INV_funcname_describe ----  invalid_test case
    4. AC_SA_s/a_funcname_describe ---- sync or async validate case
    5. AC_OT_funcname_describe ---- other action case

2. loop_case:
    * LOOP_modulename_funcname_describe

3. multithread_case:
    * MTH_modulename_funcname_describe

4. perf_case:
    * PERF_modulename_funcname_describe

5. combine_case:
    TODO:

6. mem_leak_case:
    TODO:

7. safty_case:
    TODO:

8. stable_case:
    * include:  loop_case & multithread_case & perf_case & safty_case