import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

if __name__ == '__main__':
    from main import start_shiva
    start_shiva()