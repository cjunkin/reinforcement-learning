
# If I want to train an agent from scratch:

# python train_ppo_in_multiagent_env.py \
# --log-dir data/ppo_agent_multi_v9 \
# --num-processes 10 \
# --num-steps 4_000 \
# --max-steps 10_000_000


# If I want to use a pretrained model:

python train_ppo_in_multiagent_env.py \
--log-dir data/ppo_agent_enhanced_v2_multi-2 \
--num-processes 10 \
--num-steps 4_000 \
--pretrained-model-suffix iter150 \
--pretrained-model-log-dir agents/chrispark_ppo_agent_enhanced_v2 \
--opponent-agent-name example_agent \
--max-steps 10_000_000
