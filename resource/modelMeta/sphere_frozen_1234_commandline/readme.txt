There are mainly 4 resnets in sphere network.
Frozen three of them and remain one trainable.

parameter: unfrozen <type:int>
use
#python sphere_frozen_arg.py --unfrozen=2#
which means only resnet-2 is trainable.

Input:--unfrozen=x (x=1,2,3,4;else means nofrozen)
Output: sphere_unfrozen.pb
	sphere2_feed_fetch_unfrozen.txt
	sphere2_trainable_init_var_unfrozen.txt
	sphere2_trainable_var_unfrozen.txt
