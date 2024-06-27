if (NOT EXISTS ${MNN_DIRECTORY})
    message(FATAL_ERROR "When build for ENABLE_MNN_Backend=ON, must specify mnn directory for use -DMNN_DIRECTORY=XXXor set correct mnn directory")
else ()
    set(MNN_RT_LIB ${MNN_DIRECTORY}/lib/MNN.lib)
    include_directories(${MNN_DIRECTORY}/include)
endif ()


