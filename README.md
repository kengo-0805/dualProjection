# 基本設定

- 解像度はRealSenseでバウンディングボックスを生成するときの画像以外1920x1080
- 座標情報は.txtに書き出して読みこんでいる
# コードの説明（実行順）
1. キャリブレーション
    ### 説明
    カメラ・プロジェクタのキャリブレーション

    オリジナルは戸田さんのROBOMECHのコード
    ### 実行の際の注意点
      - キャリブレーション後に"z"を押してパラメータを保存する
      - キャリブレーション後にチェスボードの画面のスクリーンショットを撮る（名前:pj1.png, pj2.png）

    ### コード
    - sample_projection.py

        プロジェクタ1台目のキャリブレーション
    - sample_projection2.py

        プロジェクタ2台目のキャリbレーション

    ### 書き出す変数
    2とついているものが2台目のプロジェクタに関係するもの．
    - homography_pjcm.txt
    - homography_pjcm2.txt
    - projection_matrix.txt
    - projection_matrix2.txt
    - modelview_matrix.txt
    - modelview_matrix2.txt

1. 人形の位置取得
    ### 説明
    人形の位置にバウンディングボックスを生成し，その角の座標を.txtに書き出すコード
    ### 注意点
    人形と同じくらいの距離に物を置かないようにする
    ### コード
    - realsense_distance.py
    ### 書き出す変数
    - bb_cord.txt
      
      [[左上x, 左下x, 右下x, 右上x],[左上y, 左下y, 右下y, 右上y]]で入っている．

1. 座標変換の計算
    ### 説明
    バウンディングボックスの座標から影位置を算出してそれぞれのプロジェクタの視点に計算するコード．
    
    流れとしては
    1. realsenseで取得した座標を読み込みキャリブレーションで算出したhomography_pjcmの逆行列をかけてプロジェクタ1の視点に変換する．
    1. その座標をcam2world(), plot_points_on_wall()で影の3次元座標に変換
    1. 影の座標をそれぞれのプロジェクタのuvにworld2cam()で変換する．

    bb_cord（バウンディングボックスの座標）→pj_cord（プロジェクタ1視点）→points（影の座標）→pj1sh_cord, pj2sh_cord（影座標をそれぞれの視点のuvにしたもの）
    ### コード
    - calc_bb2sh.py
    ### 書き出す変数
    [[左上x, 左上y], [左下x, 左下y], [右下x, 右下y], [右上x, 右上y]]の形で入っている．

    - pjjjj_cord.txt

    BBをプロジェクタ1視点に変換したもの
    - shadow_cord.txt
    
    影の3次元座標
    - pj1_cord.txt, pj2_cord.txt
    
    チェスボードの角の座標をそれぞれの視点に変換したもの（今はまだ使っていない）
    - pj1sh_cord.txt, pj2sh_cord.txt
    
    影の座標をそれぞれの視点に変換したもの
1. マスクの生成
    ### 説明
    キャリブレーションのときのスクリーンショットに対して座標計算で求めた影の位置を塗りつぶすコード

    塗り潰した画像がpj1_after.png, pj2_after.pngで保存され，それをプロジェクタから投影する
    ### コード
    - make_mask.py
