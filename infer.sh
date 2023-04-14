
export CUDA_VISIBLE_DEVICES=0,1,2,3

export LD_LIBRARY_PATH=./dependency/cuda-10.2/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=./dependency/cudnn_v8.1.1/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=./dependency/TensorRT-7.2.3.4/lib:$LD_LIBRARY_PATH

mode="trt_int8"
is_calibration="cal_on"
python infer_fifo.py \
    --model_file model/model.pdmodel \
    --params_file model/model.pdiparams \
    --input_path img.png \
    --output_path out_${mode}.png \
    --run_mode ${mode}
    --is_calibration ${is_calibration}