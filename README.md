# 🎬 MiniMax-H3 Turbo LoRA UNCENSORED — Google Colab Pro (A100)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/faelsete/minimax-h3-turbo-colab/blob/main/minimax_h3_turbo_a100.ipynb)

Ambiente 100% turnkey e automatizado para rodar o modelo **MiniMax-H3 Turbo LoRA UNCENSORED** no Google Colab Pro utilizando GPUs **NVIDIA A100 (40GB / 80GB)** com quantização 8-bit.

---

### 🚀 Diferenciais Desta Versão:

1. **Quantização 8-Bit Automática (`torchao`):**
   * O transformer original em 16-bit ocupa **38.5 GB** e causava estouro de memória (*CUDA Out of Memory*) na A100 de 40 GB.
   * Em 8-bit, o peso cai para **19.2 GB**, liberando mais de **20 GB de VRAM livre** para calcular atenção de sequências longas (10s a 14s) em resolução máxima vertical (`768x1344`) ou horizontal (`1344x768`).

2. **Modo `lazy` (100% Residente na GPU):**
   * Sem o atraso do CPU offloading. O modelo roda direto na VRAM da A100.

3. **Turbo LoRA Integrado (Larry / LightX):**
   * Apenas **4 a 6 passos de inferência** em vez dos 30 passos do modelo padrão, mantendo áudio e vídeo sincronizados.

4. **Zero Erros de Indentação:**
   * Toda a lógica de montagem e injeção do patch roda via scripts pré-compilados (`setup.sh` e `start.py`), sem necessidade de copiar e colar código sensível dentro de células do Jupyter.

---

### 💻 Como Usar no Google Colab:

Basta clicar no botão abaixo para abrir diretamente no Colab Pro:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/faelsete/minimax-h3-turbo-colab/blob/main/minimax_h3_turbo_a100.ipynb)

1. Selecione a GPU **A100** em *Ambiente de Execução > Alterar tipo de ambiente de execução*.
2. Clique em **Executar Tudo** (`Cmd + F9`).
3. Clique no link público gerado (`https://....gradio.live`) para abrir a interface web do MiniMax-H3 Studio.
