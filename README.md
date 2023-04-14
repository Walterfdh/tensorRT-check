# tensorRT-check
python=3.7
paddle install: wget https://paddle-inference-lib.bj.bcebos.com/2.2.2/python/Linux/GPU/x86-64_gcc8.2_avx_mkl_cuda10.2_cudnn8.1.1_trt7.2.3.4/paddlepaddle_gpu-2.2.2-cp37-cp37m-linux_x86_64.whl
cuda 10.2
cudnn 8.1.1
tensorrt 7.2.3.4

# tensorrt int8 to produce calibration table 
sh infer.sh # is_calibration="cal_on"
# tensorrt int inference
sh infere.sh # is_calibration="cal_off"