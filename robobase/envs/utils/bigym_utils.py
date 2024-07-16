from bigym.envs.reach_target import ReachTarget, ReachTargetDual, ReachTargetSingle
from bigym.envs.move_plates import MovePlate, MoveTwoPlates
from bigym.envs.cupboards import CupboardsOpenAll, CupboardsCloseAll, WallCupboardOpen, WallCupboardClose, DrawerTopOpen, DrawerTopClose, DrawersAllOpen, DrawersAllClose
from bigym.envs.dishwasher import DishwasherOpen, DishwasherClose, DishwasherOpenTrays, DishwasherCloseTrays
from bigym.envs.dishwasher_cups import DishwasherLoadCups, DishwasherUnloadCups, DishwasherUnloadCupsLong
from bigym.envs.dishwasher_cutlery import DishwasherLoadCutlery, DishwasherUnloadCutlery, DishwasherUnloadCutleryLong
from bigym.envs.dishwasher_plates import DishwasherLoadPlates, DishwasherUnloadPlates, DishwasherUnloadPlatesLong
from bigym.envs.pick_and_place import PutCups, TakeCups, PickBox, SaucepanToHob, StoreKitchenware, ToastSandwich, FlipSandwich, RemoveSandwich, StoreBox
from bigym.envs.manipulation import FlipCup, FlipCutlery, StackBlocks
from bigym.envs.groceries import GroceriesStoreLower, GroceriesStoreUpper

TASK_MAP = dict(
    reach_target_single=ReachTargetSingle,
    reach_target_multi_modal=ReachTarget,
    reach_target_dual=ReachTargetDual,
    stack_blocks=StackBlocks,
    move_plate=MovePlate,
    move_two_plates=MoveTwoPlates,
    flip_cup=FlipCup,
    flip_cutlery=FlipCutlery,
    dishwasher_open=DishwasherOpen,
    dishwasher_close=DishwasherClose,
    dishwasher_open_trays=DishwasherOpenTrays,
    dishwasher_close_trays=DishwasherCloseTrays,
    dishwasher_load_cups=DishwasherLoadCups,
    dishwasher_unload_cups=DishwasherUnloadCups,
    dishwasher_unload_cups_long=DishwasherUnloadCupsLong,
    dishwasher_load_cutlery=DishwasherLoadCutlery,
    dishwasher_unload_cutlery=DishwasherUnloadCutlery,
    dishwasher_unload_cutlery_long=DishwasherUnloadCutleryLong,
    dishwasher_load_plates=DishwasherLoadPlates,
    dishwasher_unload_plates=DishwasherUnloadPlates,
    dishwasher_unload_plates_long=DishwasherUnloadPlatesLong,
    drawer_top_open=DrawerTopOpen,
    drawer_top_close=DrawerTopClose,
    drawers_open_all=DrawersAllOpen,
    drawers_close_all=DrawersAllClose,
    wall_cupboard_open=WallCupboardOpen,
    wall_cupboard_close=WallCupboardClose,
    cupboards_open_all=CupboardsOpenAll,
    cupboards_close_all=CupboardsCloseAll,
    take_cups=TakeCups,
    put_cups=PutCups,
    pick_box=PickBox,
    store_box=StoreBox,
    saucepan_to_hob=SaucepanToHob,
    store_kitchenware=StoreKitchenware,
    sandwich_toast=ToastSandwich,
    sandwich_flip=FlipSandwich,
    sandwich_remove=RemoveSandwich,
    store_groceries_lower=GroceriesStoreLower,
    store_groceries_upper=GroceriesStoreUpper,
)
