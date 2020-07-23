"""
@author: Rafael Zottesso
@descprition: Gerar espectrogramas dos subdiretórios de um caminho
"""
# shutil usado para copiar arquivos
import os, sys
import winsound

def done():
    duration = 200  # milliseconds
    freq = 440  # Hz
    for i in range(5):
        winsound.Beep(freq, duration)

### Função para gerar excluir arquivos ###
def excluir_arquivos(ext, dir_origem):

    # listar todos os arquivos que estão no diretório
    conteudo = os.listdir(dir_origem)
    conteudo.sort()

    # Para cada pasta no destino...
    for subdir in conteudo:

        # verificar o subdiretório
        sub = os.path.join(dir_origem, subdir)

        # Verifica se existe algum arquivo da base e cria uma lista com os nomes ordenados
        arquivos = os.listdir(sub)

        # define os diretóriso de origem e destino para copiar os arquivos
        for a in arquivos:
            if a.endswith(".{}".format(ext)):
                os.remove(os.path.join(sub, a))


### Função para gerar espectrogramas ###
def gerar_spec(dir_origem):
    
    # listar todos os arquivos que estão no diretório
    conteudo = os.listdir(dir_origem)
    conteudo.sort()
    
    dir_destino = dir_origem + '_spec'
    # Cria o diretório de destino se não existe
    if not os.path.exists(dir_destino):
        os.mkdir(dir_destino)
    

    # Para cada pasta no destino...
    for especie in conteudo:

        # verificar o subdiretório
        dir_especie = os.path.join(dir_origem, especie)
        dir_especie_destino = os.path.join(dir_destino, especie)
        
        # Cria o diretório de destino se não existe
        if not os.path.exists(dir_especie_destino):
            os.mkdir(dir_especie_destino)

        print("Buscando arquivos de {}...".format(dir_especie))
        print("Salvando arquivos em {}...".format(dir_especie_destino))

        # Verifica se existe algum arquivo da base e cria uma lista com os nomes ordenados
        arquivos = os.listdir(dir_especie)

        # define os diretóriso de origem e destino para copiar os arquivos
        for a in arquivos:

            # Cria o nome e caminho do arquivo
            arquivo_original = os.path.join(dir_especie, a)
            arquivo_novo = os.path.join(dir_especie_destino, a)

            print("Gerando espectrograma de {}".format(arquivo_original))

            # Arquivo de saída (img do espectrograma)
            img = arquivo_novo.replace(".wav",".png")

            # Gera os spectrogramas para os arquivos
            # -m = monocromático
            # -r = raw
            # -X = densidade de pixels/s
            comando = "sox " + arquivo_original + " -n rate 22000 spectrogram -z 60 -X 27 -y 694 -r -o " + img
            # comando = "sox " + arquivo_original + " -n rate 22000 spectrogram -z 60 -x 400 -y 348 -r -o " + img
            # Executa o comando
            os.system(comando)
            
    # Toca um som ao terminar
    done()


#####################################################################
# Diretórios usados para consulta da base
dir_origem = './base_audio'

gerar_spec(dir_origem)
# excluir(".wav", dir_origem)
