# 🎬 MiniMax-H3 Turbo LoRA UNCENSORED — Google Colab Pro (A100)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/faelsete/minimax-h3-turbo-colab/blob/main/minimax_h3_turbo_a100.ipynb)

Ambiente 100% turnkey e automatizado para rodar o modelo **MiniMax-H3 Turbo LoRA UNCENSORED** no Google Colab Pro utilizando GPUs **NVIDIA A100 (40GB / 80GB)** com quantização 8-bit.

---

### 🚀 Diferenciais Desta Versão:

1. **Quantização 8-Bit Automática (`torchao`):**
   * O transformer original em 16-bit ocupa **66 GB** e causava estouro de memória (*CUDA Out of Memory*) na A100 de 40 GB.
   * Em 8-bit, o peso cai para **33 GB**, rodando liso na A100 (40 GB) sem nenhum erro de memória.

2. **Gerenciamento Inteligente de VRAM (`offload` + `expandable_segments`):**
   * Descarrega os VAEs na CPU durante o denoise e os aloca dinamicamente na decodificação final.
   * Estabilidade absoluta garantida em 100% das execuções.

3. **Turbo LoRA Integrado (Larry / LightX):**
   * Apenas **4 a 6 passos de inferência** em vez dos 30 passos do modelo padrão, gerando vídeo e áudio sincronizados em menos de 2 minutos e meio!

4. **Zero Erros de Indentação & Download Fácil:**
   * Toda a injeção roda via scripts pré-compilados (`setup.sh` e `start.py`), sem necessidade de mexer em código no Colab.
   * Os vídeos gerados são automaticamente disponibilizados no Studio web, salvos em `/content/` para acesso rápido e prontos para sincronizar no Google Drive.

---

### 💻 Como Usar no Google Colab:

Basta clicar no botão abaixo para abrir diretamente no Colab Pro:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/faelsete/minimax-h3-turbo-colab/blob/main/minimax_h3_turbo_a100.ipynb)

1. Selecione a GPU **A100** em *Ambiente de Execução > Alterar tipo de ambiente de execução*.
2. Clique em **Executar Tudo** (`Cmd + F9`).
3. Clique no link público gerado (`https://....gradio.live`) para abrir a interface web do MiniMax-H3 Studio.
4. Digite seu prompt, faça upload das imagens de início/fim (se desejar) e clique em **Generate**.
