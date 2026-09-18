import sys

from cinebot_ml.experiments.cold_start import main as cold_start_main
from cinebot_ml.experiments.matrix import main as matrix_main


if len(sys.argv) > 1 and sys.argv[1] == "cold-start":
    raise SystemExit(cold_start_main(sys.argv[2:]))
raise SystemExit(matrix_main())
