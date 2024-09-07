# taskname="novel_object"
rootpath="datasets"
# task=$taskname
# steps=(step_0 step_1 step_2 final_state)
DEVICEID=0
DEVICENUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
ratio=0.1
features=(0)
((GPUWAIT=2*$DEVICENUM))
SwitchDevice(){
    ((DEVICEID=$DEVICEID+1))
    if [ $DEVICEID == $GPUWAIT ];then
        wait
    fi
    ((DEVICEID=$DEVICEID%$GPUWAIT))
    ((port=$RANDOM+1000))
    echo "switch device"
}
# objects=("pick up the pink doll with two red ears." "pick up the yellow doll with two black ears.")
objects=("pink doll." "yellow doll.")
object_names=("pickdoll" "yellowdoll")
# objects=("upper left shelf" 'right shelf')
# object_names=("upper_left_shelf" 'right_shelf')
# objects=("Red duck." "Blue object." "Yellow duck.")
# object_names=("red_duck" "blue_duck" 'yellow_duck')
# objects=("duck")
# object_names=("duck")
# objects=("blue")
# object_names=("blue")
# objects=("Yellow doll." "Gray doll.")
# echo ${objects[1]}
for i  in $(seq 1 1);do
    # object=${objects[$i]}
    # for episode in $(ls logs/$taskname);do
        episode=episode_0
        echo $episode
        echo $DEVICEID
        # SwitchDevice
        CUDA_VISIBLE_DEVICES=$DEVICEID python visualize_lang_feat.py --object "${objects[$i]}" \
            --obj-name ${object_names[$i]} \
            --feature-path logs/${task}/${episode}/chkpnt7000.pth \
            --top-k 18000 &
        SwitchDevice
    # done
    # echo $i
done
wait
# for object in ${objects[@]};do
#     echo $object
# done
# for episode in $(ls logs/$taskname);do

# done
