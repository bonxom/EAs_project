import os
import sys
from pathlib import Path

from hifo import hifo
from hifo.utils.getParas import Paras

# --- 9router configuration ---
LLM_ENDPOINT = "http://localhost:20128/v1"   # 9router OpenAI-compatible base URL
LLM_MODEL = "ag/gemini-3.7-flash-low"


def load_api_key() -> str:
    """Resolve the 9router API key without hardcoding it.

    Priority:
      1. NINE_ROUTER_API_KEY environment variable
      2. api_key.txt next to this script (first non-empty, non-comment line)
      3. Interactive prompt (only when running in a terminal)
    """
    key = os.environ.get("NINE_ROUTER_API_KEY", "").strip()
    if key:
        return key

    key_file = Path(__file__).with_name("api_key.txt")
    if key_file.exists():
        for line in key_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line

    if sys.stdin.isatty():
        import getpass
        return getpass.getpass("Enter 9router API key: ").strip()

    return ""


API_KEY = load_api_key()
if not API_KEY:
    print(">> No 9router API key found.")
    print("   Do one of the following:")
    print("     - paste it into api_key.txt next to this script, or")
    print("     - export NINE_ROUTER_API_KEY=<your-key>, or")
    print("     - run this script in a terminal to be prompted.")
    sys.exit(1)

# Parameter initilization #
paras = Paras() 

# Set parameters #
paras.set_paras(method = "hifo",    # ['ael','hifo']
                problem = "tsp_construct", #['tsp_construct','bp_online']
                llm_api_endpoint = LLM_ENDPOINT,
                llm_api_key = API_KEY,
                llm_model = LLM_MODEL,
                ec_pop_size = 8, # number of samples in each population
                ec_n_pop = 8,  # number of populations
                exp_n_proc = 4,  # multi-core parallel
                exp_debug_mode = False)

# initilization
evolution = hifo.EVOL(paras)
# run 
evolution.run()
