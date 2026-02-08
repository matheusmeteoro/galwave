import os
import time
import configparser
import numpy as np
import pywt
from pathlib import Path
from astropy.io import fits
import warnings
from PIL import Image

# Ignorar avisos de cabeçalho do Astropy
warnings.filterwarnings('ignore', category=UserWarning, append=True)

class FITSWaveletProcessor:
    def __init__(self, config_path='config.ini'):
        self.config = configparser.ConfigParser()
        if not os.path.exists(config_path):
            self._create_default_config(config_path)
        self.config.read(config_path)
        self._load_config()

    def _create_default_config(self, path):
        self.config['PATHS'] = {'input_folder': 'DATA', 'output_folder': 'PROCESSED'}
        self.config['PARAMETERS'] = {'wavelet_type': 'bior3.3', 'decomposition_level': '4'}
        self.config['OUTPUT'] = {'format': 'BOTH', 'save_mode': 'all_layers'}
        with open(path, 'w') as f: self.config.write(f)

    def _load_config(self):
        try:
            self.input_dir = Path(self.config['PATHS']['input_folder'])
            self.output_dir = Path(self.config['PATHS']['output_folder'])
            self.wavelet = self.config['PARAMETERS']['wavelet_type']
            self.level = int(self.config['PARAMETERS']['decomposition_level'])
        except KeyError as e:
            print(f"Erro no Config: {e}")

    def _save_image(self, data, path_prefix, header=None, resize_to=None):
        """Salva FITS científico e PNG com escala Logarítmica e Redimensionamento."""
        try:
            # 1. Salvar FITS (Dados originais para sua tese)
            fits_path = f"{path_prefix}.fits"
            hdu = fits.PrimaryHDU(data, header=header)
            hdu.writeto(fits_path, overwrite=True)

            # 2. Preparar PNG (Visualização de alta qualidade)
            png_path = f"{path_prefix}.png"
            
            # Limpeza de NaNs e valores negativos para o Log
            clean_data = np.nan_to_num(data, nan=0.0)
            
            # Escalonamento Logarítmico: Essencial para ver estruturas fracas (discos/braços)
            # Adicionamos um pequeno offset para evitar log(0)
            log_data = np.log10(clean_data - np.min(clean_data) + 1.0)
            
            d_min, d_max = np.min(log_data), np.max(log_data)
            
            if d_max - d_min > 1e-6:
                rescaled = (log_data - d_min) / (d_max - d_min) * 255
            else:
                rescaled = np.zeros_like(log_data)

            # Criar imagem
            img = Image.fromarray(rescaled.astype(np.uint8))

            # Redimensionar para o tamanho original da galáxia se solicitado
            # Isso evita que as camadas L3, L4 pareçam "merda" (pixeladas)
            if resize_to:
                # Usamos BILINEAR para suavizar a visualização das camadas pequenas
                img = img.resize((resize_to[1], resize_to[0]), Image.BILINEAR)

            img.save(png_path)
            
        except Exception as e:
            print(f"Erro ao salvar: {e}")

    def process_file(self, filepath):
        try:
            with fits.open(filepath) as hdul:
                data = hdul[0].data if hdul[0].data is not None else hdul[1].data
                header = hdul[0].header.copy()
            
            if data is None: return False, "FITS vazio"
            if data.ndim > 2: data = data[0, :, :]

            orig_shape = data.shape
            galaxy_name = filepath.stem
            galaxy_folder = self.output_dir / galaxy_name
            galaxy_folder.mkdir(parents=True, exist_ok=True)

            # Decomposição Wavelet
            coeffs = pywt.wavedec2(data, self.wavelet, level=self.level)
            
            # Salvar Aproximação (Estrutura Principal / Bojo)
            self._save_image(coeffs[0], str(galaxy_folder / f"{galaxy_name}_core_approx"), 
                             header, resize_to=orig_shape)

            # Salvar Camadas de Detalhes
            for i, (h, v, d) in enumerate(coeffs[1:]):
                level_idx = self.level - i
                base_path = str(galaxy_folder / f"L{level_idx}")
                
                self._save_image(h, f"{base_path}_horizontal", header, resize_to=orig_shape)
                self._save_image(v, f"{base_path}_vertical", header, resize_to=orig_shape)
                self._save_image(d, f"{base_path}_diagonal", header, resize_to=orig_shape)

            return True, "Sucesso"

        except Exception as e:
            return False, str(e)

    def run_batch(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        files = list(self.input_dir.glob('*.fits')) + list(self.input_dir.glob('*.fit'))
        
        print(f"--- Processando {len(files)} imagens com Escala Log e Resize ---")
        for fpath in files:
            print(f"-> {fpath.name}", end=" ")
            success, msg = self.process_file(fpath)
            print(f"[{msg}]")

if __name__ == "__main__":
    processor = FITSWaveletProcessor()
    processor.run_batch()