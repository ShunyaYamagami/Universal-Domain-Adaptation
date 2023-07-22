function process_args {
    declare -A args

    local gpu_i=$1
    # local exec_num=$2
    local tmux_session=""
    local config="office-train-config.yaml"
    # local config="officehome-train-config.yaml"
    
    local params=$(getopt -n "$0"  -l tmux: -- "$@")
    eval set -- "$params"

    while true; do
        case "$1" in
            --tmux)
                tmux_session="$2"
                shift 2
                ;;
            --)
                shift
                break
                ;;
            *)
                echo "不明な引数: $1" >&2
                return 1
                ;;
        esac
    done
    echo "gpu_i: $gpu_i"
    echo "tmux: $tmux_session"
    echo -e ''  # (今は使っていないが)改行文字は echo コマンドに -e オプションを付けて実行した場合にのみ機能する.

    ###################################################
    COMMAND="conda activate universal_da "
    # for n in $(seq 0 6)
    # do
    #     COMMAND+=" && CUDA_VISIBLE_DEVICES=$gpu_i  exec_num=$n  python  main.py  --config $config"
    # done
    # COMMAND+=" && CUDA_VISIBLE_DEVICES=$gpu_i  exec_num=0  python  main.py  --config $config"
    COMMAND+=" && CUDA_VISIBLE_DEVICES=$gpu_i  exec_num=3  python  main.py  --config $config"
    COMMAND+=" && CUDA_VISIBLE_DEVICES=$gpu_i  exec_num=4  python  main.py  --config $config"

    COMMAND+=" && conserve $gpu_i"
    ###################################################
    
    ###################################################
    ###### 実行. 
    echo $COMMAND
    echo ''

    if [ -n "$tmux_session" ]; then
        # 第3引数が存在する場合の処理. tmux内で実行する. $tmux_sessionはtmuxのセッション名.
        tmux -2 new -d -s $tmux_session
        tmux send-key -t $tmux_session.0 "$COMMAND" ENTER
    else
        # 第3引数が存在しない場合の処理. そのまま実行.
        eval $COMMAND
    fi

}
####################################################
########## Verify the number of arguments ##########
# 最初の3つの引数をチェック
if [ "$#" -lt 1 ]; then
    echo "エラー: 引数が足りません。最初の1つの引数は必須です。" >&2
    return 1
fi

########## Main ##########
process_args "$@"