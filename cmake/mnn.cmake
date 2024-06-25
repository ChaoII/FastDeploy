# get RKNPU2_URL
set(MNN_DIRECTORY "D:/MNN/install")
set(MNN_VERSION "2.9.1")

# set path
# include lib
if (EXISTS ${MNN_DIRECTORY})
    set(MNN_RT_LIB ${MNN_DIRECTORY}/lib/MNN.lib)
    include_directories(${MNN_DIRECTORY}/include)
else ()
    message(FATAL_ERROR "[mnn.cmake] MNN_DIRECTORY does not exist.")
endif ()


