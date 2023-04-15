from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Event, Pipe
import time
import sys
import os
import logging
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf

from common_utils.data_loader_utils import load_mimic_cxr_bySection
from common_utils.common_utils import remove_dirs,check_and_move_file,check_and_remove_dirs

logger = logging.getLogger()
module_path = os.path.dirname(__file__)
config_path = os.path.join(os.path.dirname(module_path), "config")
START_EVENT = Event()

def batch_processing(file_name,model_path,section_name,root_path,sid_list,pid_list):
    START_EVENT.wait()
    removable = False
    try:
        pid = pid_list[sid_list.index(file_name.rstrip(".csv"))]
        pid_3char = pid[0:3]
        new_path = os.path.join(model_path,pid_3char,pid,section_name)
        file_source_path = os.path.join(root_path,file_name)
        file_destination_path = os.path.join(new_path,file_name)
        check_and_move_file(file_source_path, file_destination_path)
        removable = True
    except ValueError:
        pass
    return removable

@hydra.main(version_base=None, config_path=config_path, config_name="linguistic_preprocessing")
def main(config):
    print(OmegaConf.to_yaml(config))

    if config.clear_history:
        logger.info("Deleted history dirs: %s", config.output.base_dir)
        remove_dirs(config.output.base_dir)

    # Load data
    section_name_cfg = config.name_style.mimic_cxr.section_name
    output_section_cfg = config.output.section
    input_path = config.input.path
    logger.info("Loading mimic-cxr section data from %s", config.output.base_dir)
    data_size, pid_list, sid_list, section_list = load_mimic_cxr_bySection(input_path, output_section_cfg, section_name_cfg)
    
    for root_path, dir_list, file_list in os.walk(config.output.base_dir):
        logger.info("Re-organizing folder: %s", root_path)
        model_path = os.path.dirname(root_path)
        section_name = os.path.basename(root_path)
        removable = False
        all_task = []
        with ProcessPoolExecutor(max_workers=int(config.spacy.multiprocess_workers)//2+1) as executor:
            # Submit tasks
            for file_name in tqdm(file_list):
                all_task.append(executor.submit(batch_processing,file_name,model_path,section_name,root_path,sid_list,pid_list))
            
            # Notify tasks to start
            START_EVENT.set()

            # When a submitted task finished, the output is received here.
            if all_task:
                with tqdm(total=len(file_list)) as pbar:
                    for future in as_completed(all_task):
                        removable = future.result()
                        pbar.update(1)

            executor.shutdown(wait=True, cancel_futures=False)
            START_EVENT.clear()
        
        logger.info("Delete folder %s", root_path)
        check_and_remove_dirs(root_path,removable)
            

if __name__ == "__main__":
    sys.argv.append("linguistic_preprocessing@_global_=mimic_cxr")
    main()  # pylint: disable=no-value-for-parameter
