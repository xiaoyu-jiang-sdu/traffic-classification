python tc_train.py --train_epoch=10 \
--checkpoints='./checkpoints/Tor-8cls/' \
--data_dir='./data/Tor/' --num_labels=8 \
--label_names_json='./mapper/Tor.json' \
--prompt_domain=1 \
--content='The Tor datasets includes encrypted traffic with Tor, involving various aspects of life'