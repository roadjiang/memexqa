


python train_embedding.py --model=lr_embedding_q --train_dir=/Users/lujiang/run --data_path=/Users/lujiang/data/memex_dataset/exp/lr_embedding_tr.p --photo_feat=/Users/lujiang/data/memex_dataset/exp/photo_feat.p
python train_embedding.py --model=lr_embedding_q_i --train_dir=/Users/lujiang/run --data_path=/Users/lujiang/data/memex_dataset/exp/lr_embedding_tr.p --photo_feat=/Users/lujiang/data/memex_dataset/exp/photo_feat.p
python train_embedding.py --model=lr_embedding_q_i --train_dir=/Users/lujiang/run/pca --data_path=/Users/lujiang/data/memex_dataset/exp/lr_embedding_tr.p --photo_feat=/Users/lujiang/data/memex_dataset/exp/photo_feat_pca.p


python train_bow.py --model=bow --train_dir=/Users/lujiang/run --data_path=/Users/lujiang/data/memex_dataset/exp/bow_tr.p


python train_lstm.py --model=lstm_q --train_dir=/Users/lujiang/run --data_path=/Users/lujiang/data/memex_dataset/exp/lr_embedding_tr.p --photo_feat=/Users/lujiang/data/memex_dataset/exp/photo_feat.p
python train_lstm.py --model=lstm_q_i --train_dir=/Users/lujiang/run --data_path=/Users/lujiang/data/memex_dataset/exp/lr_embedding_tr.p --photo_feat=/Users/lujiang/data/memex_dataset/exp/photo_feat.p
python train_lstm.py --model=lstm_q_i --train_dir=/Users/lujiang/run/pca --data_path=/Users/lujiang/data/memex_dataset/exp/lr_embedding_tr.p --photo_feat=/Users/lujiang/data/memex_dataset/exp/photo_feat_pca.p


python test_embedding.py --model=lr_embedding_q --train_dir=/Users/lujiang/run --test_dir=/Users/lujiang/run --data_path=/Users/lujiang/data/memex_dataset/exp/lr_embedding_tr.p --photo_feat=/Users/lujiang/data/memex_dataset/exp/photo_feat.p --ground_truth_file=/Users/lujiang/data/memex_dataset/exp/qa_album.p
python test_embedding.py --model=lr_embedding_q_i --train_dir=/Users/lujiang/run --test_dir=/Users/lujiang/run --data_path=/Users/lujiang/data/memex_dataset/exp/lr_embedding_tr.p --photo_feat=/Users/lujiang/data/memex_dataset/exp/photo_feat.p --ground_truth_file=/Users/lujiang/data/memex_dataset/exp/qa_album.p
python test_embedding.py --model=lr_embedding_q_i --train_dir=/Users/lujiang/run/pca --test_dir=/Users/lujiang/run/pca --data_path=/Users/lujiang/data/memex_dataset/exp/lr_embedding_tr.p --photo_feat=/Users/lujiang/data/memex_dataset/exp/photo_feat_pca.p --ground_truth_file=/Users/lujiang/data/memex_dataset/exp/qa_album.p
python test_bow.py --model=bow --train_dir=/Users/lujiang/run --test_dir=/Users/lujiang/run --data_path=/Users/lujiang/data/memex_dataset/exp/bow_tr.p --ground_truth_file=/Users/lujiang/data/memex_dataset/exp/qa_album.p
python test_lstm.py --model=lstm_q --train_dir=/Users/lujiang/run --test_dir=/Users/lujiang/run --data_path=/Users/lujiang/data/memex_dataset/exp/lr_embedding_tr.p --photo_feat=/Users/lujiang/data/memex_dataset/exp/photo_feat.p --ground_truth_file=/Users/lujiang/data/memex_dataset/exp/qa_album.p
python test_lstm.py --model=lstm_q_i --train_dir=/Users/lujiang/run --test_dir=/Users/lujiang/run --data_path=/Users/lujiang/data/memex_dataset/exp/lr_embedding_tr.p --photo_feat=/Users/lujiang/data/memex_dataset/exp/photo_feat.p --ground_truth_file=/Users/lujiang/data/memex_dataset/exp/qa_album.p
