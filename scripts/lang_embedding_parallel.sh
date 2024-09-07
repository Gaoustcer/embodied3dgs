DEVICE=0
DEVICENUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
((GPUWAIT=3*$DEVICENUM))
SwitchDevice(){
    ((DEVICEID=$DEVICEID+1))
    if [ $DEVICEID == $GPUWAIT ];then
        wait
    fi
    ((DEVICE=$DEVICEID%$DEVICENUM))
    ((port=$RANDOM+1000))
    echo "switch device"
}
# tasks=(lamp_on open_box reach_target)
# for task in ${tasks[*]};do
rootpath=datasets/$task/

for episode in $(ls $rootpath);do
    echo $episode
    dataset_path=$rootpath/$episode
    for img_name in $(ls $dataset_path/images);do
        CUDA_VISIBLE_DEVICES=${DEVICE} python pixelwise_clipfeature.py --dataset_path $dataset_path --image_name $img_name &
        SwitchDevice
        # echo $DEVICE
        # echo $img_name
    done
    # ((DEVICE=$DEVICE+1))
    # SwitchDevice
done
    # rootpath=datasets/rlbench/gaussian_dataset/$task
    # steps=(step_0 step_1 step_2 final_state)
echo "loop over"
wait
