# !/bin/bash

python /root/MAREC/coding_beifen/entropy研究/entry.py --target_domain 0 --task_run 'targetbook' --gpu 0 --task '2' &
python /root/MAREC/coding_beifen/entropy研究/entry1.py --target_domain 0 --task_run 'targetbook' --gpu 1 --task '2' &
python /root/MAREC/coding_beifen/entropy研究/entry2.py --target_domain 0 --task_run 'targetbook' --gpu 1 --task '2' &




# python /root/MAREC/coding_beifen/entropy研究/entry.py --target_domain 0 --task_run 'targetbook' --gpu 2 --task '2' &
# python /root/MAREC/coding_beifen/entropy研究/entry.py --target_domain 1 --task_run 'targetcd' --gpu 0 --task '1' &
# python /root/MAREC/coding_beifen/entropy研究/entry.py --target_domain 2 --task_run 'targetmv' --gpu 0 --task '1' &
# python /root/MAREC/coding_beifen/entropy研究/entry.py --target_domain 3 --task_run 'targetel' --gpu 1 --task '1' &
# python /root/MAREC/coding_beifen/entropy研究/entry1.py --target_domain 0 --task_run 'targetbook' --gpu 0 --task '2' &
# python /root/MAREC/coding_beifen/entropy研究/entry1.py --target_domain 1 --task_run 'targetcd' --gpu 1 --task '1' &
# python /root/MAREC/coding_beifen/entropy研究/entry1.py --target_domain 2 --task_run 'targetmv' --gpu 2 --task '1' &
# python /root/MAREC/coding_beifen/entropy研究/entry1.py --target_domain 3 --task_run 'targetel' --gpu 1 --task '1' &
# python /root/MAREC/coding_beifen/entropy研究/entry2.py --target_domain 0 --task_run 'targetbook' --gpu 1 --task '2' &
# python /root/MAREC/coding_beifen/entropy研究/entry2.py --target_domain 1 --task_run 'targetcd' --gpu 2 --task '1' &
# python /root/MAREC/coding_beifen/entropy研究/entry2.py --target_domain 2 --task_run 'targetmv' --gpu 2 --task '1' &
# python /root/MAREC/coding_beifen/entropy研究/entry2.py --target_domain 3 --task_run 'targetel' --gpu 0 --task '1' &

#python /root/MAREC/coding_beifen/MAPPO基层/entry.py --root "/root/autodl-tmp/root/autodl-tmp/multidomain" --target_domain 1 --task_run 'targetcd' --gpu 1 --task '1' &
# python /root/MAREC/coding_beifen/MAPPO基层/entry.py --root "/root/autodl-tmp/root/autodl-tmp/multidomain" --target_domain 2 --task_run 'targetmv' --gpu 2 --task '1' &
# python /root/MAREC/coding_beifen/MAPPO基层/entry.py --root "/root/autodl-tmp/root/autodl-tmp/multidomain" --target_domain 3 --task_run 'targetel' --gpu 2 --task '1' &