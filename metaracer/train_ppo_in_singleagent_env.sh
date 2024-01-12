# If I want to train an agent from scratch:

python train_ppo_in_singleagent_env.py \
--log-dir data/ppo_agent_enhanced_v2_2 \
--num-processes 10 \
--num-steps 4_000 \
--max-steps 10_000_000 \
--pretrained-model-log-dir agents/chrispark_ppo_agent_enhanced \
--pretrained-model-suffix final
