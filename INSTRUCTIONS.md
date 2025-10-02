
# 在 Ubuntu 伺服器上執行 GNN+XGBoost 模型的完整教學

本文檔將指導您如何在一個乾淨的 Ubuntu 環境中，設定必要的 Python 環境並執行詐騙偵測模型。

---

## 步驟一：前置準備

請確保您已經將競賽資料與模型腳本上傳到伺服器。

假設您的伺服器目錄結構如下：

```
/home/user/race/              <-- 您的專案根目錄
├── submission_template.csv
├── 資料欄位說明.csv
└── 初賽資料/                 <-- 主要工作目錄
    ├── acct_transaction.csv
    ├── acct_alert.csv
    └── run_gnn_xgb_pipeline.py  <-- 我們的主腳本
```

**重要**：請將 `run_gnn_xgb_pipeline.py` 檔案放在 `初賽資料/` 目錄下，並從 `初賽資料/` 目錄中執行所有指令。

---

## 步驟二：安裝 Python 與相依套件

我們將使用 `pip` 來安裝所有必要的 Python 函式庫。建議使用 Python 3.8 或更高版本。

1.  **更新套件列表**

    ```bash
    sudo apt-get update
    ```

2.  **安裝 pip** (如果您的環境沒有的話)

    ```bash
    sudo apt-get install -y python3-pip
    ```

3.  **安裝核心函式庫**

    ```bash
    pip3 install pandas numpy scikit-learn tqdm xgboost
    ```

4.  **安裝 PyTorch (CPU 或 GPU 版本)**

    *   **選項 A：安裝 CPU 版本 (如果您的伺服器沒有 NVIDIA GPU)**

        ```bash
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```

    *   **選項 B：安裝 GPU 版本 (推薦，如果伺服器有 NVIDIA GPU 且已安裝 CUDA)**
        請根據您伺服器上的 CUDA 版本，到 [PyTorch 官網](https://pytorch.org/get-started/locally/) 查詢對應的安裝指令。例如，如果您的 CUDA 版本是 11.8，指令會是：

        ```bash
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```

5.  **安裝 PyTorch Geometric (PyG)**

    這是執行 GNN 的關鍵函式庫。它的安裝依賴於您剛剛安裝的 PyTorch 版本。

    ```bash
    pip3 install torch_geometric
    ```

---

## 步驟三：執行模型管線

一切準備就緒後，您只需要一個指令就可以執行完整的流程。

1.  **切換到工作目錄**

    ```bash
    cd /path/to/your/project/race/初賽資料/
    ```
    > 請將 `/path/to/your/project/` 替換成您實際的路徑。

2.  **執行 Python 腳本**

    ```bash
    python3 run_gnn_xgb_pipeline.py
    ```

---

## 步驟四：預期輸出與結果

執行後，您會在終端機上看到類似以下的進度資訊：

```
使用設備: cuda

--- 階段一：特徵工程 ---
正在載入資料...
訓練集截止日期 (train_cutoff_date): ...
測試集截止日期 (test_cutoff_date): ...
建立圖結構與節點嵌入...
訓練 GNN...
GNN Training: 100%|██████████| 10/10 [00:15<00:00,  1.52s/it]
生成節點嵌入...
為截止日期 ... 計算靜態特徵...
...
特徵工程完成！

--- 階段二：XGBoost 模型訓練與預測 ---
進行 5-Fold 交叉驗證訓練...
--- Fold 1 ---
[0]	eval-logloss:...
[500]	eval-logloss:...
...

在 OOF 預測上尋找最佳 F1 門檻值...
Threshold Search: 100%|██████████| 80/80 [00:00<00:00, ...it/s]
最佳 F1 分數 (OOF): ...
最佳門檻值: ...

生成提交檔案...

=======================================
流程成功結束！
提交檔案 'submission.csv' 已生成。
預測出 ... 個警示帳戶。
=======================================
```

流程結束後，在 `初賽資料/` 目錄下就會產生一個名為 `submission.csv` 的檔案，這就是您可以直接上傳到競賽平台的最終結果。

---

## 常見問題

*   **CUDA out of memory**: 如果您在執行 GNN 或 XGBoost 時遇到 GPU 記憶體不足的錯誤，可以嘗試編輯 `run_gnn_xgb_pipeline.py` 腳本，調小 `GNN_HIDDEN_DIM` 的數值，或是在 XGBoost 設定中移除 `tree_method='gpu_hist'` 來改用 CPU 訓練。
*   **找不到檔案**: 請再三確認您執行 `python3` 指令時，所在的目錄是 `初賽資料/`。

祝您競賽順利！
