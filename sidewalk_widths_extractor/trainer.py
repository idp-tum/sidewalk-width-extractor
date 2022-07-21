import json
import os
from typing import Dict, Optional, Union

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sidewalk_widths_extractor.modules import BaseModule
from sidewalk_widths_extractor.typing import _DEVICE, _EPOCH_RESULT, _PATH, _STEP_RESULT
from sidewalk_widths_extractor.utilities import (
    Category,
    Metric,
    create_path_with_timestamp,
    get_device,
    mkdir,
    save_writer_figures,
    save_writer_scalars,
)


class Trainer:
    """
    Trainer

    Attributes:
        device: the associated device.
        log_dir: the directory where the log files and checkpoints are stored.
        trained: whether the module is trained or not.
        resumed: whether the module is resumed or not.
    """

    def __init__(
        self,
        log_folder: _PATH = "logs",
        log_comment: Optional[str] = None,
        override_log_dir: Optional[str] = None,
        progress_bar: bool = True,
        benchmark: bool = False,
        deterministic: bool = False,
        transfer_results_to_cpu: bool = False,
    ) -> None:
        """
        Trainer

        Args:
            log_folder (optional): directory path to store logs.
                Defaults to "logs".
            log_comment (optional): comment to be added to log filenames as a postfix.
                Defaults to None.
            override_log_dir (optional): path to a directory where logs will be saved.
                if override_log_dir is given, log_folder and log_comment parameters will be overwritten.
                Defaults to None.
            progress_bar: whether to use Tqdm progress bar.
                Defaults to True.
            benchmark: whether to open pytorch benchmark functionality.
                Defaults to False.
            deterministic: whether to open pytorch determinstic functionality.
                Defaults to False.
            transfer_results_to_cpu: whether to transfer epoch results to cpu.
                Defaults to False.
        """
        self._progress_bar_disabled: bool = not progress_bar
        self._transfer_results_to_cpu: bool = transfer_results_to_cpu

        self._benchmark = benchmark
        torch.backends.cudnn.benchmark = benchmark
        self._deterministic = deterministic
        torch.use_deterministic_algorithms(deterministic)

        if override_log_dir:
            self._log_comment = None
            self._log_path = override_log_dir
            self._log_checkpoint_path = os.path.join(self._log_path, "checkpoints")
        else:
            self._log_comment = log_comment
            self._log_path = create_path_with_timestamp(log_folder, postfix=log_comment)
            self._log_checkpoint_path = os.path.join(self._log_path, "checkpoints")

        self._start_epoch_idx: int = 1
        self._curr_epoch_idx: int = 1

        self._trained: bool = False
        self._resumed: bool = False

        self._device: _DEVICE = None
        self._module: BaseModule = None
        self._writer: SummaryWriter = None

    @property
    def trained(self) -> bool:
        return self._trained

    @property
    def resumed(self) -> bool:
        return self._resumed

    @property
    def device(self) -> _DEVICE:
        return self._device if self._device else get_device()

    @property
    def log_dir(self) -> str:
        return os.path.abspath(self._log_path)

    def fit(
        self,
        module: BaseModule,
        dataloader: DataLoader,
        validate_dataloader: Optional[DataLoader] = None,
        validate_every_n_epoch: int = 1,
        max_epochs: int = 1,
        max_steps: Optional[int] = None,
        checkpoint_path: Optional[Union[Dict[str, _PATH], _PATH]] = None,
        save_every_n_epoch: Optional[int] = None,
        save_settings: bool = True,
        save_scalars: bool = False,
        save_figures: bool = False,
    ) -> None:
        """
        Main fitting/training function.

        Args:
            module: to-be-used module.
            dataloader: training data loader instance.
            validate_dataloader (optional): validatuon data loader instance.
                 Defaults to None.
            validate_every_n_epoch: frequency of validating.
                Defaults to 1.
            max_epochs: total number of epoch to run.
                Defaults to 1.
            max_steps (optional): total number of step to run.
                Defaults to None.
            checkpoint_path (optional): path to the checkpoint file(s).
                Defaults to None.
            save_every_n_epoch (optional): frequency of saving between validation epochs.
                Defaults to None.
            save_settings: whether to save general trainer and module settings as yaml file.
                Defaults to True.
            save_scalars: whether tensorboard scalars are saved as a csv file.
                Defaults to True.
            save_figures: whether to save tensorboard figures.
                Defaults to True.
        """
        assert isinstance(module, BaseModule)
        assert isinstance(dataloader, DataLoader)

        self._module = module
        if self._module.resumed:
            self._resumed = True
            if self._module.curr_epoch_idx:
                self._start_epoch_idx = self._module.curr_epoch_idx + 1
                self._curr_epoch_idx = self._start_epoch_idx

        if checkpoint_path:
            epoch = self._module.load(checkpoint_path)
            if epoch:
                self._start_epoch_idx = epoch + 1
                self._curr_epoch_idx = self._start_epoch_idx
                self._module.curr_epoch_idx = epoch
                max_epochs = max_epochs + epoch
            self._resumed = True

        assert self._start_epoch_idx <= max_epochs and self._curr_epoch_idx <= max_epochs
        assert not max_steps or max_steps >= 1

        if validate_dataloader:
            assert isinstance(validate_dataloader, DataLoader)
            assert validate_every_n_epoch >= 1

        self._trained = True
        self._setup()

        self._module.on_start()

        for epoch_idx in range(self._start_epoch_idx, max_epochs + 1):
            self._curr_epoch_idx = epoch_idx
            self._module.curr_epoch_idx = epoch_idx

            # main training loop
            self._module.on_train_epoch_start(self._curr_epoch_idx)
            train_epoch_results = self._train_epoch(dataloader, epoch_idx, max_steps)
            self._module.on_train_epoch_end(train_epoch_results, epoch_idx)

            # validation if activated
            if validate_dataloader and epoch_idx % validate_every_n_epoch == 0:
                self._module.on_val_epoch_start(self._curr_epoch_idx)
                val_epoch_results = self._validate_epoch(validate_dataloader, epoch_idx, max_steps)
                self._module.on_val_epoch_end(val_epoch_results, epoch_idx)

            # saving checkpoint if necessary
            if (save_every_n_epoch and (epoch_idx % save_every_n_epoch == 0)) or (
                epoch_idx == max_epochs
            ):
                self._module.save(epoch_idx)

        self._module.on_end()

        if save_settings:
            self.save_settings()
        if save_scalars:
            save_writer_scalars(self._log_path)
        if save_figures:
            save_writer_figures(self._log_path)

    def tune(
        self,
        module: BaseModule,
        dataloader: DataLoader,
        validate_dataloader: DataLoader,
        epoch_idx: Optional[int] = None,
    ) -> _EPOCH_RESULT:
        assert isinstance(module, BaseModule)
        assert isinstance(dataloader, DataLoader)
        assert isinstance(validate_dataloader, DataLoader)

        self._module = module
        self._curr_epoch_idx = epoch_idx

        self._trained = True
        if epoch_idx == 1:
            self._setup()

        self._module.on_train_epoch_start(epoch_idx)
        train_epoch_results = self._train_epoch(dataloader, epoch_idx)
        self._module.on_train_epoch_end(train_epoch_results, epoch_idx)

        self._module.on_val_epoch_start(epoch_idx)
        val_epoch_results = self._validate_epoch(validate_dataloader, epoch_idx)
        self._module.on_val_epoch_end(val_epoch_results, epoch_idx)

        return {k: v.values for k, v in val_epoch_results.items()}

    def validate(
        self,
        dataloader: DataLoader,
        module: Optional[BaseModule] = None,
        max_steps: Optional[int] = None,
        checkpoint_path: Optional[Union[Dict[str, _PATH], _PATH]] = None,
    ) -> _EPOCH_RESULT:
        """
        Nain validation function.

        Args:
            dataloader: validation data loader instance.
            module (optional): to-be-used module. if not given, trainer will use the prevously provided module.
                Defaults to None.
            max_steps (optional): total number of steps to run. Defaults to None.
        """
        if module:
            assert isinstance(module, BaseModule)
            self._module = module
        else:
            assert isinstance(self._module, BaseModule)
            assert self._trained or self._resumed

        if checkpoint_path:
            epoch = self._module.load(checkpoint_path)
            if epoch:
                self._start_epoch_idx = epoch + 1
                self._curr_epoch_idx = self._start_epoch_idx
            self._resumed = True

        self._setup(ignore_writer=True)

        self._module.on_val_epoch_start()
        epoch_results = self._validate_epoch(dataloader, None, max_steps)
        self._module.on_val_epoch_end(epoch_results)

        return {k: v.values for k, v in epoch_results.items()}

    def test(
        self,
        dataloader: DataLoader,
        module: Optional[BaseModule] = None,
        max_steps: Optional[int] = None,
        checkpoint_path: Optional[Union[Dict[str, _PATH], _PATH]] = None,
    ) -> _EPOCH_RESULT:
        """
        Main test function.

        Args:
            dataloader: test data loader instance.
            module (optional): to-be-used module. if not given, trainer will use the prevously provided module.
                Defaults to None.
            max_steps (optional): total number of steps to run. Defaults to None.
        """
        if module:
            assert isinstance(module, BaseModule)
            self._module = module
        else:
            assert isinstance(self._module, BaseModule)
            assert self._trained or self._resumed

        if checkpoint_path:
            epoch = self._module.load(checkpoint_path)
            if epoch:
                self._start_epoch_idx = epoch + 1
                self._curr_epoch_idx = self._start_epoch_idx
            self._resumed = True

        self._setup(ignore_writer=True)

        self._module.on_test_epoch_start()
        epoch_results = self._test_epoch(dataloader, None, max_steps)
        self._module.on_test_epoch_end(epoch_results)

        return {k: v.values for k, v in epoch_results.items()}

    def _train_epoch(
        self,
        dataloader: DataLoader,
        epoch_idx: int,
        max_steps: Optional[int] = None,
    ) -> _EPOCH_RESULT:
        """
        Training epoch function.

        Args:
            dataloader: training data loader instance.
            epoch_idx: current epoch index.
            max_steps (optional): total number of steps to run. Defaults to None.

        Returns:
            _EPOCH_RESULT: the result of the training epoch.
        """
        pb = self._init_progress_bar(dataloader, Category.TRAINING, epoch_idx)

        epoch_results: _EPOCH_RESULT = {}

        torch.set_grad_enabled(True)
        for step_idx, batch in enumerate(pb):
            if max_steps and step_idx > max_steps - 1:
                break
            self._module.on_train_step_start(epoch_idx, step_idx)
            step_results = self._module.train_step(batch, step_idx, epoch_idx)
            self._module.on_train_step_end(step_results, epoch_idx, step_idx)

            self._accumulate_results(epoch_results, step_results, Category.TRAINING)

        return epoch_results

    def _validate_epoch(
        self,
        dataloader: DataLoader,
        epoch_idx: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> _EPOCH_RESULT:
        """
        Validation epoch function.

        Args:
            dataloader: validation data loader instance.
            epoch_idx: current epoch index.
            max_steps (optional): total number of steps to run. Defaults to None.

        Returns:
            _EPOCH_RESULT: the result of the validation epoch.
        """
        pb = self._init_progress_bar(dataloader, Category.VALIDATION, epoch_idx)

        epoch_results: _EPOCH_RESULT = {}

        torch.set_grad_enabled(False)
        for step_idx, batch in enumerate(pb):
            if max_steps and step_idx > max_steps - 1:
                break
            self._module.on_val_step_start(epoch_idx, step_idx)
            step_results = self._module.validate_step(batch, step_idx, epoch_idx)
            self._module.on_val_step_end(step_results, epoch_idx, step_idx)

            self._accumulate_results(epoch_results, step_results, Category.VALIDATION)

        return epoch_results

    def _test_epoch(
        self,
        dataloader: DataLoader,
        epoch_idx: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> _EPOCH_RESULT:
        """
        Test epoch function.

        Args:
            dataloader: test data loader instance.
            epoch_idx: current epoch index.
            max_steps (optional): total number of steps to run. Defaults to None.

        Returns:
            _EPOCH_RESULT: the result of the test epoch.
        """
        pb = self._init_progress_bar(dataloader, Category.TEST, epoch_idx)

        epoch_results: _EPOCH_RESULT = {}

        torch.set_grad_enabled(False)
        for step_idx, batch in enumerate(pb):
            if max_steps and step_idx > max_steps - 1:
                break

            self._module.on_test_step_start(epoch_idx, step_idx)
            step_results = self._module.test_step(batch, step_idx, epoch_idx)
            self._module.on_test_step_end(step_results, epoch_idx, step_idx)

            self._accumulate_results(epoch_results, step_results, Category.TEST)

        return epoch_results

    def _setup(self, ignore_writer: bool = False) -> None:
        """
        General Setup function.
        """
        self._device = self._module.device
        self._module.trained = self._trained
        self._module.resumed = self._resumed
        self._module.checkpoint_folder_path = self._log_checkpoint_path
        self._module.log_path = self._log_path
        if not ignore_writer:
            mkdir(self._log_checkpoint_path)
            if not self._writer:
                self._writer = SummaryWriter(self._log_path)
            self._module.writer = self._writer

    def save_settings(self) -> None:
        settings = {
            "run": {
                "run_id": os.path.basename(os.path.normpath(self._log_path)),
                "device": str(self._device),
                "resumed": self._resumed,
                "starting_epoch_idx": self._start_epoch_idx,
                "ending_epoch_idx": self._curr_epoch_idx,
                "pytorch": {
                    "benchmark": self._benchmark,
                    "deterministic": self._deterministic,
                },
            }
        }
        settings["module"] = self._module.get_settings()

        with open(os.path.join(self._log_path, "settings.json"), "w") as file:
            json.dump(settings, file, indent=4, sort_keys=False)

    def _init_progress_bar(
        self, dataloader: DataLoader, category: Category, epoch_idx: Optional[int] = None
    ) -> tqdm:
        """
        Initialize a tqdm progress bar with the given arguments.

        Args:
            dataloader: dataloader instance.
            category: trainer category.
            epoch_idx (optional): current epoch number.

        Returns:
            tqdm: tqdm progress bar instance.
        """
        if category == Category.TRAINING:
            title = "Training"
        elif category == Category.VALIDATION:
            title = "Validating"
        else:
            title = "Testing"

        desc = ""

        if epoch_idx:
            desc += f"[{epoch_idx}] "

        if self._log_comment:
            desc += f"{self._log_comment} - {title}"
        else:
            desc += f"{title}"

        return tqdm(
            dataloader,
            desc=desc,
            ncols=0,
            disable=self._progress_bar_disabled,
        )

    def _accumulate_results(
        self, epoch_results: _EPOCH_RESULT, step_results: _STEP_RESULT, category: Category
    ) -> None:
        """
        Accumulate epoch results by appending new step results.

        Args:
            epoch_results: existing epoch results.
            step_results: new step results.
            category: metric category.
        """
        if not epoch_results:
            for name, value in step_results.items():
                epoch_results[name] = Metric(name, category, value, self._transfer_results_to_cpu)
        else:
            for name, metric in epoch_results.items():
                metric.update(step_results[name])
