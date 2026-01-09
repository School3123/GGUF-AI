import streamlit as st
import os
from llama_cpp import Llama

# ページ設定
st.set_page_config(page_title="GGUF AI Chat", page_icon="🤖")

st.title("🤖 GGUF Model Chat App")
st.write("GGUFファイルをアップロードして、AIと会話しましょう。")

# サイドバーでモデルのアップロード
st.sidebar.header("モデル設定")
uploaded_file = st.sidebar.file_uploader("GGUFファイルをアップロード", type=["gguf"])

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm" not in st.session_state:
    st.session_state.llm = None

# モデルのロード処理
if uploaded_file is not None:
    # 一時ファイルとして保存（llama.cppはファイルパスが必要なため）
    model_path = f"./temp_{uploaded_file.name}"
    
    # ファイルがまだ保存されていない、または新しいファイルの場合のみ保存
    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Saved: {uploaded_file.name}")
        
        # モデルをロード（キャッシュクリアのためにセッションリセット）
        st.session_state.llm = None
        st.session_state.messages = []

    # LLMの初期化
    if st.session_state.llm is None:
        try:
            with st.spinner("モデルを読み込んでいます... (数分かかる場合があります)"):
                # n_ctxはコンテキストサイズ。必要に応じて調整してください（例: 2048, 4096）
                st.session_state.llm = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    verbose=False
                )
            st.sidebar.success("モデル読み込み完了！")
        except Exception as e:
            st.error(f"モデルの読み込みに失敗しました: {e}")

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# チャット入力
if prompt := st.chat_input("メッセージを入力してください..."):
    # ユーザーの入力を表示
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.llm:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # 推論実行
            with st.spinner("考え中..."):
                stream = st.session_state.llm.create_chat_completion(
                    messages=st.session_state.messages,
                    stream=True
                )
                
                for output in stream:
                    if len(output['choices']) > 0:
                        delta = output['choices'][0]['delta']
                        if 'content' in delta:
                            full_response += delta['content']
                            message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.warning("先に左側のサイドバーからGGUFファイルをアップロードしてください。")

# 一時ファイルのクリーンアップ（必要に応じて実装。今回はシンプルにするため省略）
