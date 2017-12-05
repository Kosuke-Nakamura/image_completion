シーンの大局的かつ局所的な整合性を考慮した画像補完を行う
元の論文: http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/ja/
*.sh はremoteにはあげてない


com_{train, updater}
	補完器をMSEのみを用いて学習する
	train: メインのスクリプト
	updater: パラメータ更新
	
dis_train, dis_updater
	識別器のみを学習する
	train: メインのスクリプト
	updater: パラメータ更新

gl_train, gl_updater
	補完器，識別器を同時に学習する
	train: メインのスクリプト
	updater: パラメータ更新

gl_net
	補完ネットワーク，識別ネットワークが記述してある

gl_fnctions
	ネットワーク中で用いる各種関数が記述されている

gl_visualize
	学習結果としてランダムに補完された画像を出力するためのtrainerのextension
	サンプルを作るのにも使われる

gl_dataset
	データセットとして指定したディレクトリの画像を読み出す
	イテレータにこれを与えることでエポックの度に画像を配列として読み出して渡す
	全画像の画素の平均値を計算したりもする

gl_make_samples
	補完モデルとデータセットをあたえることでマスクしてそれを保管した画像を出力
	同じ配置で元画像，マスクを付与した画像も出力できる

train_{com, dis, train}.sh
	各種引数を設定して{com, dis, gl}_train.pyを実行するシェルスクリプト
	com -> dis -> gl
	の順で実行することで全体の学習が完了する


train_mnist.sh
	MNISTを使って補完ネットワークを学習する
	MNISTのうち１万枚を使う

make_{mnist, celeba}_sample.sh
	指定した補完ネットワークを用いてMNISTかcelebaの補完を行うサンプルを生成する
	
