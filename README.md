# AI_Doll

A local voice-interaction prototype that starts a session through a **face-recognition access gate**, uses **Whisper** for speech recognition, combines **RAG** for retrieval-augmented response generation, and uses **XTTS v2** to synthesize and play voice responses.

---

## 1. Project Capabilities Overview览

- Recognizes a specified user through the camera, using `imgs/biden.jpg` as the default reference image.
- Starts and stops audio recording through Enter key control.
- Uses Whisper (`turbo`) to convert speech into text.
- Calls the RAG pipeline in `src/query.py` to generate responses.
- Uses Coqui TTS `xtts_v2` to synthesize Chinese speech based on a reference voice.
- Provides interaction and status prompts in the terminal through `rich`.

---

## 2. Actual Code Structure

```text
AI_Doll/
├── README.md
├── requirement.txt
├── localchat.py                 # Main program: face recognition + recording + STT + RAG + TTS
├── imgs/
│   ├── biden.jpg                # Default face reference image
│   └── obama.jpg
├── target_voice/
│   ├── xinmeng_audio.wav        # Default speaker reference audio for TTS voice cloning
│   ├── dobby.mp3
│   └── hemine.wav
└── src/
    ├── database.py              # Document splitting and vector database construction using Chroma
    ├── query.py                 # RAG query entry point: query_rag
    ├── document/
    │   └── 作品.docx            # Default knowledge-base document
    └── chroma/                  # Existing vector database directory
```

---

## 3. Runtime Workflow

```text
Start localchat.py
  -> Open the camera and perform face recognition
  -> Play the reference audio after the target user is recognized
  -> Press Enter to start recording, and press Enter again to stop
  -> Transcribe the audio with Whisper
  -> Retrieve information and generate a response through query_rag()
  -> Synthesize and play the response with XTTS
  -> Continue to the next dialogue round
```

---

## 4. Environment Setup

## 4.1 Recommended Python Version

- Python 3.10 to 3.12 is recommended. The project contains multiple dependencies, and versions that are too new or too old may cause compatibility issues.

## 4.2 Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 4.3 Install Dependencies

This repository provides a complete dependency lock file, requirement.txt, encoded in UTF-16 with BOM. It is recommended to try:

```bash
pip install -r requirement.txt
```

If you only want to run the main pipeline first, you may install the minimum required dependencies:

```bash
pip install numpy openai-whisper sounddevice rich torch TTS face_recognition opencv-python \
            langchain langchain-openai langchain-community langchain-text-splitters chromadb \
            python-dotenv docx2txt
```

> face_recognition depends on dlib. Some systems may require C/C++ build tools and CMake to be installed in advance.

---

## 5. Configuration

## 5.1 Face Recognition Configuration in localchat.py

- Default face image: imgs/biden.jpg
- Default recognized name: Reze

You may modify these values as needed:

```python
biden_image = face_recognition.load_image_file("imgs/biden.jpg")
known_face_names = ["Reze"]
```

## 5.2 Voice-Cloning Reference Audio in localchat.py

The default audio file is:

```python
"target_voice/xinmeng_audio.wav"
```

This audio is used for:
- Playing a prompt audio after face recognition succeeds.
- Serving as the speaker_wav reference for XTTS voice cloning.

## 5.3 RAG and Model Configuration in src/query.py

- Vector database directory: `src/chroma`
- Embedding：`OllamaEmbeddings(model="nomic-embed-text")`
- LLM：`ChatOpenAI(model="gpt-4o")`
- Custom API Base URL：`https://api.guidaodeng.com/v1`

Please make sure that:
1. Ollama is available on the local machine and nomic-embed-text has been pulled.
2. Your OpenAI-compatible API key environment variable has been configured, usually as OPENAI_API_KEY.

---

## 6. Knowledge Base Construction

This step is optional but recommended. After updating documents under src/document/, it is recommended to rebuild the vector database:

```bash
cd src
python database.py --reset
python database.py
```

Notes:
- --reset deletes src/chroma and rebuilds it.
- .pdf, .docx, and .txt documents are supported.

---

## Start the Project

Run the following command from the repository root directory:

```bash
python localchat.py
```

Interaction steps:
1. Pass face recognition first.
2. Press Enter once to start recording.
3. Press Enter again to stop recording.
4. Wait for transcription, retrieval, and voice playback.

Press Ctrl + C to exit.

---

## 8. Troubleshooting

### 8.1 The Camera Cannot Open or the Face Cannot Be Recognized

- Make sure camera permission has been granted.
- Make sure imgs/biden.jpg contains a clear frontal face.
- You may replace the target face image with a clearer high-resolution frontal image.

### 8.2 No Microphone Input

- Check whether the system default recording device is correct.
- Try closing other applications that may be using the microphone.

### 8.3 TTS Is Slow or GPU Memory Is Insufficient

- The program automatically uses CUDA if available; otherwise, it falls back to CPU.
- xtts_v2 is significantly slower on CPU, which is expected.

### RAG Call Fails

Check the following items:
- Whether the Ollama service has been started.
- Whether the nomic-embed-text model has been pulled.
- Whether the API key and base_url are available.
- Whether src/chroma has been successfully built.
