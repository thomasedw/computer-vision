import os
import cv2
from PIL import Image
from util import draw_bounding_box
from ObjectDetector import ObjectDetector
from SemanticSegmenter import Segmenter
from PyramidDisparityCalculator import PyramidDisparityCalculator
from StandardStereoDisparity import StandardDisparityCalculator
from MonoDisparityCalculator import MonoDisparityCalculator
import numpy as np
import time
import math
import matplotlib.pyplot as plt

##############################
# COMPUTER VISION COURSEWORK #
##############################

master_path_to_dataset = "C://Users/thoma/development/coursework/software_systems_and_applications/Computer Vision/implementation/TTBB-durham-02-10-17-sub10"
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"
segmentation_colors = {
    "background": (0, 0, 0,),
    "aeroplane": (128, 0, 0),
    "bicycle": (0, 128, 0),
    "bird": (128, 128, 0),
    "boat": (0, 0, 128),
    "bus": (0, 128, 128),
    "car": (128, 128, 128),
    "cat": (64, 0, 0),
    "cow": (64, 128, 0),
    "dog": (64, 128, 128),
    "motorbike": (64, 128, 128),
    "person": (192, 0, 128),
    "train": (128, 192, 0)
}

full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

black_threshold = 1

focal_length = 399.9745178222656
baseline = 0.2090607502

# For each frame we:
#   - Detect objects using YOLO
#   - Utilize semantic segmentation on detected objects to extract areas of interest
#   - Calculate the disparity map using x network
#   - Use the segmentation as a binary mask on the disparity map
#   - Calculate the mean of the area of interest for each detected object and calculate distances

object_detector = ObjectDetector(master_path_to_dataset, confidence_threshold=0.50)
object_classes = object_detector.get_classes()
segmenter = Segmenter()
max_disparity_levels = 32*8

# disparity_calculator = MonoDisparityCalculator(model_path="C://Users/thoma/development/coursework/software_systems_and_applications/Computer Vision/implementation/monodepth2-config/")
disparity_calculator = PyramidDisparityCalculator(max_disparity=max_disparity_levels, model_path='C://Users/thoma/development/coursework/software_systems_and_applications/Computer Vision/implementation/pretrained_model_KITTI2015.tar')
# disparity_calculator = StandardDisparityCalculator(max_disparity=max_disparity_levels)

# video_filenames = [
#     "1506942473.484027_L.png", #start
#     "1506942474.483193_L.png",
#     "1506942475.481834_L.png",
#     "1506942476.480930_L.png",
#     "1506942477.481815_L.png",
#     "1506942478.481487_L.png",
#     "1506942479.480726_L.png",
#     "1506942480.483420_L.png",
#     "1506942481.483936_L.png",
#     "1506942482.482131_L.png",
#     "1506942483.480862_L.png",
#     "1506942484.480963_L.png",
#     "1506942485.480259_L.png",
#     "1506942486.479530_L.png",
#     "1506942487.479214_L.png",
#     "1506942488.478824_L.png",
#     "1506942489.478285_L.png",
#     "1506942490.477398_L.png",
#     "1506942491.476786_L.png",
#     "1506942492.476647_L.png",
#     "1506942493.476187_L.png",
#     "1506942494.475828_L.png",
#     "1506942495.475615_L.png",
#     "1506942496.475553_L.png",
#     "1506942497.475560_L.png",
#     "1506942498.475580_L.png",
#     "1506942499.475533_L.png",
#     "1506942500.475414_L.png",
#     "1506942501.475423_L.png",
#     "1506942502.475483_L.png",
#     "1506942503.475460_L.png",
#     "1506942504.475323_L.png",
#     "1506942505.475215_L.png",
#     "1506942506.475136_L.png",
#     "1506942507.475112_L.png",
#     "1506942508.475117_L.png",
#     "1506942509.475122_L.png",
#     "1506942510.475119_L.png",
#     "1506942511.475133_L.png",
#     "1506942512.475150_L.png",
#     "1506942513.475180_L.png",
#     "1506942514.475191_L.png",
#     "1506942515.475229_L.png",
#     "1506942516.475195_L.png",
#     "1506942517.475163_L.png",
#     "1506942518.475202_L.png",
#     "1506942519.475168_L.png",
#     "1506942520.475073_L.png",
#     "1506942521.474985_L.png",
#     "1506942522.475027_L.png",
#     "1506942523.475169_L.png",
#     "1506942524.475219_L.png",
#     "1506942525.475251_L.png",
#     "1506942526.475234_L.png",
#     "1506942527.475220_L.png",
#     "1506942528.475263_L.png",
#     "1506942529.475284_L.png",
#     "1506942530.475244_L.png",
#     "1506942531.475245_L.png",
#     "1506942532.475181_L.png",
#     "1506942533.475109_L.png",
#     "1506942534.475102_L.png",
#     "1506942535.475090_L.png",
#     "1506942536.475079_L.png",
#     "1506942537.475077_L.png",
#     "1506942538.475113_L.png",
#     "1506942539.475152_L.png",
#     "1506942540.475228_L.png",
#     "1506942541.475200_L.png",
#     "1506942542.475174_L.png",
#     "1506942543.475237_L.png",
#     "1506942544.475300_L.png",
#     "1506942545.475430_L.png",
#     "1506942546.475550_L.png",
#     "1506942547.475639_L.png",
#     "1506942548.475714_L.png", #end of start
#     "1506942595.475435_L.png", #church street bus
#     "1506942596.475391_L.png",
#     "1506942597.475340_L.png",
#     "1506942598.475325_L.png",
#     "1506942599.475348_L.png",
#     "1506942600.475344_L.png",
#     "1506942601.475389_L.png",
#     "1506942602.475396_L.png",
#     "1506942603.475343_L.png",
#     "1506942604.475373_L.png",
#     "1506942605.475618_L.png",
#     "1506942606.475622_L.png",
#     "1506942607.475560_L.png",
#     "1506942608.475424_L.png",
#     "1506942609.475570_L.png",
#     "1506942610.475772_L.png",
#     "1506942611.475872_L.png",
#     "1506942612.475913_L.png",
#     "1506942613.475850_L.png",
#     "1506942614.475869_L.png",
#     "1506942615.475632_L.png",
#     "1506942616.475693_L.png",
#     "1506942617.475673_L.png",
#     "1506942618.475629_L.png",
#     "1506942619.475628_L.png",
#     "1506942620.475645_L.png",
#     "1506942621.475723_L.png",
#     "1506942622.475877_L.png",
#     "1506942623.476026_L.png",
#     "1506942624.476122_L.png",
#     "1506942625.476076_L.png",
#     "1506942626.475818_L.png",
#     "1506942627.475673_L.png",
#     "1506942628.475582_L.png",
#     "1506942629.475753_L.png",
#     "1506942630.475890_L.png",
#     "1506942631.475946_L.png",
#     "1506942632.475832_L.png",
#     "1506942633.475894_L.png",
#     "1506942634.475903_L.png",
#     "1506942635.476001_L.png",
#     "1506942636.476171_L.png",
#     "1506942637.476314_L.png",
#     "1506942638.476344_L.png",
#     "1506942639.476455_L.png",
#     "1506942640.476440_L.png",
#     "1506942641.476450_L.png",
#     "1506942642.476433_L.png",
#     "1506942643.476350_L.png",
#     "1506942644.476266_L.png",
#     "1506942645.476571_L.png",
#     "1506942646.476488_L.png",
#     "1506942647.476417_L.png",
#     "1506942648.476540_L.png",
#     "1506942649.476499_L.png",
#     "1506942650.476505_L.png",
#     "1506942651.476637_L.png",
#     "1506942652.476794_L.png",
#     "1506942653.476678_L.png",
#     "1506942654.476566_L.png",
#     "1506942655.476593_L.png",
#     "1506942656.476552_L.png",
#     "1506942657.476605_L.png",
#     "1506942658.476606_L.png",
#     "1506942659.476543_L.png",
#     "1506942660.476443_L.png",
#     "1506942661.476521_L.png",
#     "1506942662.476535_L.png",
#     "1506942663.476757_L.png",
#     "1506942664.476670_L.png",
#     "1506942665.476632_L.png",
#     "1506942666.476703_L.png",
#     "1506942667.476685_L.png",
#     "1506942668.476668_L.png",
#     "1506942669.476637_L.png",
#     "1506942670.477170_L.png",
#     "1506942671.477079_L.png",
#     "1506942672.477009_L.png",
#     "1506942673.477016_L.png",
#     "1506942674.476965_L.png",
#     "1506942675.476861_L.png",
#     "1506942676.476753_L.png",
#     "1506942677.476683_L.png",
#     "1506942678.476632_L.png",
#     "1506942679.476572_L.png",
#     "1506942680.476593_L.png",
#     "1506942681.476652_L.png",
#     "1506942682.476682_L.png",
#     "1506942683.476724_L.png",
#     "1506942684.476727_L.png",
#     "1506942685.476704_L.png",
#     "1506942686.476769_L.png",
#     "1506942687.476722_L.png",
#     "1506942688.476654_L.png",
#     "1506942689.476653_L.png",
#     "1506942690.476617_L.png",
#     "1506942691.476549_L.png",
#     "1506942692.476661_L.png",
#     "1506942693.476771_L.png",
#     "1506942694.477003_L.png",
#     "1506942695.478586_L.png",
#     "1506942696.480014_L.png",
#     "1506942697.480157_L.png",
#     "1506942698.480303_L.png",
#     "1506942699.478361_L.png",
#     "1506942700.478326_L.png",
#     "1506942701.477707_L.png",
#     "1506942702.476849_L.png",
#     "1506942703.476622_L.png",
#     "1506942704.476515_L.png",
#     "1506942705.476540_L.png",
#     "1506942706.476551_L.png",
#     "1506942707.476526_L.png",
#     "1506942708.476520_L.png",
#     "1506942709.476522_L.png",
#     "1506942710.476566_L.png",
#     "1506942711.476585_L.png",
#     "1506942712.476154_L.png",
#     "1506942713.476619_L.png",
#     "1506942714.476591_L.png",
#     "1506942715.476848_L.png",
#     "1506942716.476855_L.png",
#     "1506942717.476839_L.png",
#     "1506942718.476805_L.png",
#     "1506942719.476793_L.png",
#     "1506942720.476777_L.png",
#     "1506942721.476770_L.png",
#     "1506942722.476806_L.png",
#     "1506942723.476778_L.png",
#     "1506942724.476754_L.png",
#     "1506942725.476749_L.png",
#     "1506942726.476755_L.png",
#     "1506942727.476730_L.png",
#     "1506942728.476727_L.png",
#     "1506942729.476739_L.png",
#     "1506942730.476761_L.png",
#     "1506942731.476803_L.png",
#     "1506942732.476775_L.png",
#     "1506942733.476768_L.png",
#     "1506942734.476745_L.png",
#     "1506942735.476744_L.png",
#     "1506942736.476778_L.png",
#     "1506942737.476764_L.png",
#     "1506942738.476779_L.png",
#     "1506942739.476874_L.png",
#     "1506942740.476823_L.png",
#     "1506942741.476807_L.png",
#     "1506942742.476773_L.png",
#     "1506942743.476744_L.png",
#     "1506942744.476731_L.png",
#     "1506942745.476734_L.png",
#     "1506942746.476718_L.png",
#     "1506942747.476804_L.png",
#     "1506942748.477018_L.png",
#     "1506942749.476870_L.png",
#     "1506942750.476486_L.png",
#     "1506942751.476383_L.png",
#     "1506942752.476437_L.png",
#     "1506942753.476428_L.png",
#     "1506942754.476480_L.png",
#     "1506942755.476388_L.png",
#     "1506942756.476288_L.png",
#     "1506942757.476238_L.png",
#     "1506942758.476032_L.png",
#     "1506942759.475881_L.png",
#     "1506942760.475846_L.png",
#     "1506942761.475891_L.png",
#     "1506942762.475864_L.png",
#     "1506942763.475785_L.png",
#     "1506942764.475746_L.png",
#     "1506942765.475656_L.png",
#     "1506942766.475879_L.png",
#     "1506942767.476345_L.png",
#     "1506942768.476444_L.png",
#     "1506942769.475901_L.png",
#     "1506942770.475915_L.png",
#     "1506942771.476037_L.png",
#     "1506942772.476260_L.png",
#     "1506942773.476423_L.png",
#     "1506942774.476786_L.png",
#     "1506942775.477553_L.png",
#     "1506942776.480631_L.png",
#     "1506942777.481345_L.png",
#     "1506942778.479167_L.png",
#     "1506942779.477724_L.png",
#     "1506942780.477349_L.png",
#     "1506942781.476897_L.png",
#     "1506942782.476581_L.png",
#     "1506942783.476583_L.png",
#     "1506942784.476787_L.png",
#     "1506942785.477064_L.png",
#     "1506942786.476894_L.png",
#     "1506942787.476912_L.png",
#     "1506942788.476933_L.png",
#     "1506942789.477163_L.png",
#     "1506942790.477355_L.png",
#     "1506942791.477258_L.png",
#     "1506942792.477428_L.png",
#     "1506942793.477749_L.png",
#     "1506942794.477844_L.png",
#     "1506942795.477751_L.png",
#     "1506942796.477520_L.png",
#     "1506942797.477606_L.png",
#     "1506942798.477520_L.png",
#     "1506942799.477478_L.png",
#     "1506942800.477502_L.png",
#     "1506942801.477495_L.png",
#     "1506942802.477454_L.png",
#     "1506942803.477449_L.png",
#     "1506942804.477428_L.png",
#     "1506942805.477429_L.png",
#     "1506942806.477399_L.png",
#     "1506942807.477344_L.png",
#     "1506942808.477228_L.png",
#     "1506942809.477216_L.png",
#     "1506942810.477136_L.png",
#     "1506942811.477003_L.png",
#     "1506942812.476805_L.png",
#     "1506942813.476504_L.png",
#     "1506942814.476364_L.png",
#     "1506942815.476285_L.png",
#     "1506942816.476358_L.png",
#     "1506942817.476407_L.png",
#     "1506942818.476491_L.png",
#     "1506942819.476735_L.png",
#     "1506942820.476864_L.png",
#     "1506942821.476878_L.png",
#     "1506942822.476838_L.png",
#     "1506942823.477103_L.png",
#     "1506942824.476623_L.png",
#     "1506942825.476899_L.png",
#     "1506942826.478000_L.png",
#     "1506942827.479384_L.png",
#     "1506942828.480218_L.png",
#     "1506942829.480903_L.png",
#     "1506942830.481546_L.png",
#     "1506942831.482083_L.png", #end of church street
#     "1506942930.483557_L.png", #start of saddler street
#     "1506942931.485515_L.png",
#     "1506942932.485130_L.png",
#     "1506942933.484695_L.png",
#     "1506942934.485674_L.png",
#     "1506942935.484018_L.png",
#     "1506942936.481877_L.png",
#     "1506942937.480641_L.png",
#     "1506942938.479666_L.png",
#     "1506942939.478676_L.png",
#     "1506942940.478307_L.png",
#     "1506942941.478070_L.png",
#     "1506942942.477675_L.png",
#     "1506942943.478446_L.png",
#     "1506942944.478514_L.png",
#     "1506942945.478549_L.png",
#     "1506942946.480088_L.png",
#     "1506942947.480370_L.png",
#     "1506942948.479527_L.png",
#     "1506942949.478592_L.png",
#     "1506942950.478921_L.png",
#     "1506942951.479304_L.png",
#     "1506942952.479769_L.png",
#     "1506942953.480995_L.png",
#     "1506942954.481669_L.png",
#     "1506942955.481876_L.png",
#     "1506942956.481856_L.png",
#     "1506942957.481889_L.png",
#     "1506942958.481944_L.png",
#     "1506942959.482054_L.png",
#     "1506942960.482497_L.png",
#     "1506942961.483067_L.png",
#     "1506942962.483417_L.png",
#     "1506942963.483489_L.png",
#     "1506942964.483469_L.png",
#     "1506942965.483449_L.png",
#     "1506942966.483447_L.png",
#     "1506942967.483425_L.png",
#     "1506942968.483396_L.png",
#     "1506942969.483861_L.png",
#     "1506942970.483900_L.png",
#     "1506942971.483993_L.png",
#     "1506942972.484011_L.png",
#     "1506942973.483978_L.png",
#     "1506942974.484008_L.png",
#     "1506942975.484023_L.png",
#     "1506942976.484042_L.png",
#     "1506942977.483998_L.png",
#     "1506942978.483972_L.png",
#     "1506942979.483965_L.png",
#     "1506942980.483985_L.png",
#     "1506942981.484048_L.png",
#     "1506942982.484123_L.png",
#     "1506942983.484180_L.png",
#     "1506942984.484093_L.png",
#     "1506942985.484067_L.png",
#     "1506942986.484076_L.png",
#     "1506942987.484422_L.png",
#     "1506942988.484560_L.png",
#     "1506942989.484440_L.png",
#     "1506942990.483967_L.png",
#     "1506942991.483011_L.png",
#     "1506942992.482222_L.png",
#     "1506942993.482255_L.png",
#     "1506942994.482340_L.png",
#     "1506942995.482430_L.png",
#     "1506942996.482805_L.png",
#     "1506942997.482954_L.png",
#     "1506942998.483153_L.png",
#     "1506942999.483389_L.png",
#     "1506943000.483309_L.png",
#     "1506943001.482602_L.png",
#     "1506943002.481945_L.png",
#     "1506943003.480253_L.png",
#     "1506943004.479069_L.png",
#     "1506943005.478367_L.png",
#     "1506943006.478169_L.png",
#     "1506943007.479011_L.png",
#     "1506943008.479793_L.png",
#     "1506943009.480358_L.png",
#     "1506943010.480501_L.png",
#     "1506943011.480140_L.png",
#     "1506943012.479647_L.png",
#     "1506943013.480154_L.png",
#     "1506943014.481064_L.png",
#     "1506943015.482226_L.png",
#     "1506943016.482676_L.png",
#     "1506943017.482900_L.png",
#     "1506943018.482395_L.png",
#     "1506943019.481484_L.png",
#     "1506943020.481201_L.png",
#     "1506943021.481045_L.png",
#     "1506943022.481102_L.png",
#     "1506943023.480281_L.png",
#     "1506943024.479518_L.png",
#     "1506943025.478840_L.png",
#     "1506943026.478262_L.png",
#     "1506943027.477714_L.png",
#     "1506943028.477441_L.png",
#     "1506943029.477275_L.png",
#     "1506943030.476772_L.png",
#     "1506943031.476865_L.png",
#     "1506943032.476964_L.png",
#     "1506943033.477067_L.png",
#     "1506943034.477654_L.png",
#     "1506943035.478214_L.png",
#     "1506943036.478570_L.png",
#     "1506943037.478566_L.png",
#     "1506943038.478635_L.png",
#     "1506943039.478751_L.png",
#     "1506943040.478283_L.png",
#     "1506943041.478215_L.png",
#     "1506943042.478516_L.png",
#     "1506943043.479218_L.png",
#     "1506943044.479945_L.png",
#     "1506943045.479433_L.png",
#     "1506943046.478701_L.png",
#     "1506943047.478101_L.png",
#     "1506943048.477608_L.png",
#     "1506943049.477325_L.png",
#     "1506943050.477681_L.png",
#     "1506943051.477804_L.png",
#     "1506943052.478510_L.png",
#     "1506943053.478622_L.png",
#     "1506943054.478816_L.png",
#     "1506943055.479435_L.png",
#     "1506943056.478730_L.png",
#     "1506943057.478327_L.png",
#     "1506943058.477989_L.png",
#     "1506943059.477681_L.png",
#     "1506943060.477936_L.png",
#     "1506943061.478682_L.png",
#     "1506943062.478723_L.png",
#     "1506943063.479369_L.png",
#     "1506943064.479127_L.png",
#     "1506943065.479636_L.png",
#     "1506943066.478776_L.png",
#     "1506943067.477958_L.png",
#     "1506943068.477626_L.png",
#     "1506943069.477658_L.png",
#     "1506943070.478036_L.png",
#     "1506943071.478179_L.png", #end of saddler street
#     "1506943182.491927_L.png", #start of palace green
#     "1506943183.491663_L.png",
#     "1506943184.491344_L.png",
#     "1506943185.488454_L.png",
#     "1506943186.484070_L.png",
#     "1506943187.482385_L.png",
#     "1506943188.482587_L.png",
#     "1506943189.483552_L.png",
#     "1506943190.484791_L.png",
#     "1506943191.487683_L.png",
#     "1506943192.489465_L.png",
#     "1506943193.490024_L.png",
#     "1506943194.490140_L.png",
#     "1506943195.490107_L.png",
#     "1506943196.489851_L.png",
#     "1506943197.489167_L.png",
#     "1506943198.486331_L.png",
#     "1506943199.482288_L.png",
#     "1506943200.479823_L.png",
#     "1506943201.478550_L.png",
#     "1506943202.477861_L.png",
#     "1506943203.477387_L.png",
#     "1506943204.477247_L.png",
#     "1506943205.477513_L.png",
#     "1506943206.479882_L.png",
#     "1506943207.485466_L.png",
#     "1506943208.488222_L.png",
#     "1506943209.488850_L.png",
#     "1506943210.489545_L.png",
#     "1506943211.488252_L.png",
#     "1506943212.486652_L.png",
#     "1506943213.485856_L.png",
#     "1506943214.486492_L.png",
#     "1506943215.487855_L.png",
#     "1506943216.489393_L.png",
#     "1506943217.488109_L.png",
#     "1506943218.486129_L.png",
#     "1506943219.483367_L.png",
#     "1506943220.483495_L.png",
#     "1506943221.487363_L.png",
#     "1506943222.489613_L.png",
#     "1506943223.485551_L.png",
#     "1506943224.482398_L.png",
#     "1506943225.481240_L.png",
#     "1506943226.481507_L.png",
#     "1506943227.487228_L.png",
#     "1506943228.490407_L.png",
#     "1506943229.490273_L.png",
#     "1506943230.488128_L.png",
#     "1506943231.485679_L.png",
#     "1506943232.485051_L.png",
#     "1506943233.484296_L.png",
#     "1506943234.483327_L.png",
#     "1506943235.483002_L.png",
#     "1506943236.483081_L.png",
#     "1506943237.484055_L.png",
#     "1506943238.485607_L.png",
#     "1506943239.487053_L.png",
#     "1506943240.489420_L.png",
#     "1506943241.495907_L.png",
#     "1506943242.499015_L.png",
#     "1506943243.498485_L.png",
#     "1506943244.496525_L.png",
#     "1506943245.495262_L.png",
#     "1506943246.493722_L.png",
#     "1506943247.492020_L.png",
#     "1506943248.492903_L.png",
#     "1506943249.493796_L.png",
#     "1506943250.494835_L.png",
#     "1506943251.495983_L.png",
#     "1506943252.496762_L.png",
#     "1506943253.498011_L.png",
#     "1506943254.498476_L.png",
#     "1506943255.498760_L.png",
#     "1506943256.496952_L.png",
#     "1506943257.493912_L.png",
#     "1506943258.492807_L.png",
#     "1506943259.491100_L.png",
#     "1506943260.487874_L.png",
#     "1506943261.483305_L.png",
#     "1506943262.483068_L.png",
#     "1506943263.483855_L.png",
#     "1506943264.484323_L.png",
#     "1506943265.484880_L.png",
#     "1506943266.485062_L.png",
#     "1506943267.484256_L.png",
#     "1506943268.481909_L.png",
#     "1506943269.479636_L.png",
#     "1506943270.478415_L.png",
#     "1506943271.478050_L.png",
#     "1506943272.477667_L.png",
#     "1506943273.478823_L.png",
#     "1506943274.480046_L.png",
#     "1506943275.481323_L.png",
#     "1506943276.482032_L.png",
#     "1506943277.481542_L.png",
#     "1506943278.479735_L.png",
#     "1506943279.479337_L.png",
#     "1506943280.479154_L.png",
#     "1506943281.479221_L.png",
#     "1506943282.479159_L.png",
#     "1506943283.479115_L.png",
#     "1506943284.479196_L.png",
#     "1506943285.479182_L.png",
#     "1506943286.479559_L.png",
#     "1506943287.480274_L.png",
#     "1506943288.481077_L.png",
#     "1506943289.481828_L.png",
#     "1506943290.482413_L.png",
#     "1506943291.482076_L.png",
#     "1506943292.482023_L.png",
#     "1506943293.482141_L.png",
#     "1506943294.481197_L.png",
#     "1506943295.480194_L.png",
#     "1506943296.480095_L.png",
#     "1506943297.480525_L.png",
#     "1506943298.481170_L.png",
#     "1506943299.483906_L.png",
#     "1506943300.483767_L.png",
#     "1506943301.484539_L.png",
#     "1506943302.485197_L.png",
#     "1506943303.488101_L.png",
#     "1506943304.487298_L.png",
#     "1506943305.483630_L.png",
#     "1506943306.480891_L.png",
#     "1506943307.479183_L.png",
#     "1506943308.478296_L.png",
#     "1506943309.478026_L.png",
#     "1506943310.478065_L.png",
#     "1506943311.478149_L.png",
#     "1506943312.478230_L.png",
#     "1506943313.478306_L.png",
#     "1506943314.478484_L.png",
#     "1506943315.478544_L.png",
#     "1506943316.478629_L.png",
#     "1506943317.478361_L.png",
#     "1506943318.478324_L.png",
#     "1506943319.478333_L.png",
#     "1506943320.478523_L.png",
#     "1506943321.478588_L.png",
#     "1506943322.478582_L.png",
#     "1506943323.481043_L.png",
#     "1506943324.486596_L.png",
#     "1506943325.488827_L.png",
#     "1506943326.489326_L.png",
#     "1506943327.487867_L.png",
#     "1506943328.487324_L.png",
#     "1506943329.484434_L.png",
#     "1506943330.482479_L.png",
#     "1506943331.480819_L.png",
#     "1506943332.479282_L.png",
#     "1506943333.478757_L.png",
#     "1506943334.478421_L.png",
#     "1506943335.478405_L.png",
#     "1506943336.478556_L.png",
#     "1506943337.478737_L.png",
#     "1506943338.478855_L.png",
#     "1506943339.478716_L.png",
#     "1506943340.478578_L.png",
#     "1506943341.478461_L.png",
#     "1506943342.478196_L.png",
#     "1506943343.478088_L.png",
#     "1506943344.479537_L.png",
#     "1506943345.483645_L.png",
#     "1506943346.486767_L.png",
#     "1506943347.486852_L.png",
#     "1506943348.486382_L.png",
#     "1506943349.485853_L.png",
#     "1506943350.485278_L.png",
#     "1506943351.484269_L.png",
#     "1506943352.483117_L.png",
#     "1506943353.482729_L.png",
#     "1506943354.483612_L.png",
#     "1506943355.484391_L.png",
#     "1506943356.483454_L.png",
#     "1506943357.483161_L.png",
#     "1506943358.481531_L.png",
#     "1506943359.479532_L.png",
#     "1506943360.478645_L.png",
#     "1506943361.478838_L.png",
#     "1506943362.478131_L.png",
#     "1506943363.477878_L.png",
#     "1506943364.477940_L.png",
#     "1506943365.477936_L.png",
#     "1506943366.477849_L.png",
#     "1506943367.477840_L.png",
#     "1506943368.477780_L.png",
#     "1506943369.477726_L.png",
#     "1506943370.477754_L.png",
#     "1506943371.477841_L.png",
#     "1506943372.477890_L.png",
#     "1506943373.477930_L.png",
#     "1506943374.477964_L.png",
#     "1506943375.477978_L.png",
#     "1506943376.478009_L.png",
#     "1506943377.477985_L.png",
#     "1506943378.478012_L.png",
#     "1506943379.478001_L.png",
#     "1506943380.478003_L.png",
#     "1506943381.477995_L.png",
#     "1506943382.478034_L.png",
#     "1506943383.478046_L.png",
#     "1506943384.478058_L.png",
#     "1506943385.478059_L.png",
#     "1506943386.478073_L.png",
#     "1506943387.478074_L.png",
#     "1506943388.478081_L.png",
#     "1506943389.478106_L.png",
#     "1506943390.478103_L.png",
#     "1506943391.478084_L.png",
#     "1506943392.478107_L.png",
#     "1506943393.478088_L.png",
#     "1506943394.478099_L.png",
#     "1506943395.478101_L.png",
#     "1506943396.478123_L.png",
#     "1506943397.478124_L.png",
#     "1506943398.478124_L.png",
#     "1506943399.478433_L.png",
#     "1506943400.478733_L.png" #end of palace green
# ]

test_filenames = [
    "1506942473.484027_L.png", #start in carpark
    "1506942480.483420_L.png", #carpark closer truck
    "1506942488.478824_L.png", #exit carpark close person
    "1506942516.475195_L.png", #exit science site person on road
    "1506942580.476202_L.png", #behind bus #1
    "1506942581.476067_L.png", #behind bus #2
    "1506942582.475816_L.png", #behind bus #3
    "1506942583.475576_L.png", #behind bus #4
    "1506942584.475481_L.png", #behind bus #5
    "1506942585.475465_L.png", #behind bus #6
    "1506942586.475038_L.png", #behind bus #7
    "1506942587.475154_L.png", #behind bus #8
    "1506942588.475210_L.png", #behind bus #9
    "1506942589.475214_L.png", #behind bus #10
    "1506942590.475215_L.png", #behind bus #11
    "1506942591.475326_L.png", #behind bus #12
    "1506942592.475323_L.png", #behind bus #13
    "1506942593.475299_L.png", #behind bus #14
    "1506942594.475307_L.png", #behind bus #15
    "1506942722.476806_L.png", #stationary behind bus around corner
    "1506942868.476841_L.png", #van with person on side
    "1506942900.476373_L.png", #motorbike
    "1506942962.483417_L.png", #person in-front of coach
    "1506943033.477067_L.png", #lots of people varying lighting
    "1506943061.478682_L.png", #test image on brief
    "1506943206.479882_L.png", #dark conditions people
    "1506943303.488101_L.png", #people with motion blur
    "1506943404.478846_L.png", #palace green people near w dogs
    "1506943477.488606_L.png", #exit palace green people near
    "1506943559.485561_L.png", #people down Saddler St.
    "1506943572.482979_L.png", #people in market square
    "1506943490.479040_L.png", #bright white van near
    "1506943635.486954_L.png", #traffic whilst parked
    "1506943762.479924_L.png", #cars in shadows
]

total_objects = 0
left_file_list = sorted(os.listdir(full_path_directory_left))
# hist_ctr = 0
for left_filename in left_file_list:
    if left_filename in test_filenames:
        # hist_ctr += 1
        right_filename = left_filename.replace("_L", "_R")
        full_path_filename_left = os.path.join(full_path_directory_left, left_filename)
        full_path_filename_right = os.path.join(full_path_directory_right, right_filename)

        if '.png' in left_filename:
            original_image = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
            left_masked_image = original_image.copy()

        start_preprocessing_time = time.time()
        # Ellipse masks out the car bonnet
        cv2.ellipse(left_masked_image, (int(left_masked_image.shape[1] / 2), left_masked_image.shape[0]), (left_masked_image.shape[0], 130), 0, 180, 360, (0, 0, 0), -1)
        left_masked_image_lab = cv2.cvtColor(left_masked_image, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(left_masked_image_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        lab_planes[0] = clahe.apply(lab_planes[0])
        left_masked_image_lab = cv2.merge(lab_planes)
        left_masked_image_lab = cv2.cvtColor(left_masked_image_lab, cv2.COLOR_LAB2BGR)

        # cv2.imshow("left_masked_image_lab", left_masked_image_lab)
        # cv2.waitKey(5000)
        # Detect objects
        start_object_detection_time = time.time()
        [class_ids, confidences, boxes] = object_detector.detect_objects(left_masked_image)
        total_objects += len(class_ids)
        end_object_detection_time = time.time()

        # Verify that right_image exists
        if (os.path.isfile(full_path_filename_right)):
            right_masked_image = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
            cv2.ellipse(right_masked_image, (int(left_masked_image.shape[1] / 2), left_masked_image.shape[0]), (left_masked_image.shape[0], 130), 0, 180, 360, (0, 0, 0), -1)
            right_masked_image_lab = cv2.cvtColor(right_masked_image, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(right_masked_image_lab)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
            lab_planes[0] = clahe.apply(lab_planes[0])
            right_masked_image_lab = cv2.merge(lab_planes)
            right_masked_image = cv2.cvtColor(right_masked_image_lab, cv2.COLOR_LAB2BGR)

            # cv2.imshow("corrected_right_masked_image", right_masked_image)
        end_preprocessing_time = time.time()
        # Calculate disparity
        start_disparity_calc_time = time.time()
        calculated_disparity = disparity_calculator.calculate_disparity(left_masked_image_lab, right_masked_image)
        # calculated_disparity = disparity_calculator.calculate_disparity(left_masked_image)
        # Zero-out and black areas in the right image (e.g. car bonnet)
        # calculated_disparity[np.any(right_masked_image == [0, 0, 0], axis=-1)] = 0
        # Multiply disparity to resonable (visible) values
        calculated_disparity = (calculated_disparity * (max_disparity_levels / 256)).astype('uint16')
        
        # cv2.imshow("calculated_disparity", calculated_disparity)
        # cv2.waitKey(500000000)
        # calculated_disparity[calculated_disparity < 20] = 0

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
        # calculated_disparity = clahe.apply(calculated_disparity)
        # cv2.imshow("contrast calculated disparity", calculated_disparity)
        # cv2.waitKey(50000)

        cv2.ellipse(calculated_disparity, (int(left_masked_image.shape[1] / 2), left_masked_image.shape[0]), (left_masked_image.shape[0], 130), 0, 180, 360, (0, 0, 0), -1)
        end_disparity_calc_time = time.time()
        # plt.hist((calculated_disparity).ravel(), 255, [1,256], alpha=0.5, label=str(hist_ctr))
        
        # if hist_ctr >= 8:
        #     plt.legend(loc='upper right')
        #     plt.show()

        average_segmentation_time = []
        
        # Loop through each detected object
        for i in range(0, len(class_ids)):
            if object_classes[class_ids[i]] in ['car', 'bus', 'truck', 'person', 'dog', 'traffic light', 'bicycle', 'motorbike']:
                segmentation_map = None

                if boxes[i][0] < 0:
                    boxes[i][0] = 0
                elif boxes[i][0] > left_masked_image.shape[1]:
                    boxes[i][0] = left_masked_image.shape[1]
                
                if boxes[i][1] < 0:
                    boxes[i][1] = 0
                elif boxes[i][1] > left_masked_image.shape[0]:
                    boxes[i][1] = left_masked_image.shape[0]

                if boxes[i][1] + boxes[i][3] > left_masked_image.shape[0]:
                    boxes[i][3] = left_masked_image.shape[0] - boxes[i][1]
                
                if boxes[i][0] + boxes[i][2] > left_masked_image.shape[1]:
                    boxes[i][2] = left_masked_image.shape[1] - boxes[i][0]

                if object_classes[class_ids[i]] in segmentation_colors:
                    object_colour = segmentation_colors[object_classes[class_ids[i]]]       
                    # Segment the detected object
                    start_segmentation_time = time.time()
                    try:
                        segmentation_map = segmenter.segment_image(Image.fromarray(left_masked_image[boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]]))
                        # Resize segmentation map to original bounding box found from YOLO
                        segmentation_map = cv2.resize(segmentation_map, (boxes[i][2], boxes[i][3]), interpolation = cv2.INTER_AREA)
                        end_segmentation_time = time.time()
                        average_segmentation_time.append(end_segmentation_time - start_segmentation_time)
                    except ValueError:
                        pass

                # Get disparity map of the YOLO bounding box
                disparity_map_of_object = calculated_disparity[boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]]

                # If the segmentation map has failed to pickup objects, or the incorrect object, we fallback to using the mean of the bouding box of YOLO
                if segmentation_map is None or np.sum(segmentation_map) < black_threshold or disparity_map_of_object[np.any(segmentation_map == object_colour, axis=-1)].size == 0:
                    average_disparity = np.mean(calculated_disparity[boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]])    
                else:
                    # Select pixels of the disparity which are in the segmentation map
                    cutoff_class_names = ["bus"]
                    if object_classes[class_ids[i]] in cutoff_class_names:
                        height, width = disparity_map_of_object.shape[0], disparity_map_of_object.shape[1]
                        disparity_map_of_object = disparity_map_of_object[height//2:height,:]
                        segmentation_map = segmentation_map[height//2:height,:]
                        disparity_segmented_pixels = disparity_map_of_object[np.any(segmentation_map == object_colour, axis=-1)]
                        # disparity_map_of_object[height//2:height,:] = 0
                        original_image[(boxes[i][1] + boxes[i][3] // 2):boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]][np.all(segmentation_map == object_colour, axis=-1)] = object_colour
                    else:
                        disparity_segmented_pixels = disparity_map_of_object[np.any(segmentation_map == object_colour, axis=-1)]
                        original_image[boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]][np.all(segmentation_map == object_colour, axis=-1)] = object_colour
                    
                    min_std = 0.5
                    max_std = 1.5
                    # disparity_segmented_pixels = disparity_segmented_pixels[(min_std * np.std(disparity_segmented_pixels) < abs(disparity_segmented_pixels - np.mean(disparity_segmented_pixels))) & (max_std * np.std(disparity_segmented_pixels) < abs(disparity_segmented_pixels - np.mean(disparity_segmented_pixels)))]
                    # disparity_map_of_object[np.any(segmentation_map != object_colour, axis=-1)] = 0

                    # disparity_segmented_pixels = disparity_map_of_object
                    # cv2.imshow("disp_seg_pixels", disparity_segmented_pixels)
                    # cv2.waitKey(500)
                    
                    # smaller objects generally have higher error in the disparity map, so we remove outliers by subtracting from the mean and comparing
                    # to the standard deviation. The thresholds vary based on the width and height of the detected objects, (larger objects being more tolerant)
                    # and smaller objects being less tolerant (penalizing higher values)
                    max_std_threshold = boxes[i][2] / 50
                    min_std_threshold = boxes[i][3] / 50
                    # print("max_std = {}, min_std = {}".format(min_std_threshold, max_std_threshold))
                    # disparity_map_of_object[abs(disparity_map_of_object - np.mean(disparity_map_of_object)) > 1.0 * np.std(disparity_map_of_object)] = 0
                    # disparity_map_of_object[abs(disparity_map_of_object - np.mean(disparity_map_of_object)) < -1.0 * np.std(disparity_map_of_object)] = 0
                    # cv2.imshow("disparity_map_of_object", disparity_map_of_object * 255)
                    # cv2.imshow("segmentation_map_of_object", segmentation_map)
                    # cv2.namedWindow("calculated_disparity", cv2.WINDOW_NORMAL)
                    # cv2.imshow("calculated_disparity", calculated_disparity * 255)
                    # cv2.waitKey(8000)
                    # disparity_segmented_pixels = disparity_segmented_pixels[abs(disparity_segmented_pixels - np.mean(disparity_segmented_pixels)) < 6 * np.std(disparity_segmented_pixels)]
                    
                    # original_image[boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]][np.any(disparity_map_of_object != 0, axis=-1)] = object_colour
                    # original_image[boxes[i][1]:boxes[i][1] + boxes[i][3], boxes[i][0]:boxes[i][0] + boxes[i][2]][np.all(segmentation_map == object_colour, axis=-1)] = object_colour

                    average_disparity = np.mean(disparity_segmented_pixels)
                estimated_distance = focal_length * (baseline / average_disparity)
                depth_error = estimated_distance ** 2 / (baseline * focal_length) * 1.1
                estimated_distance += depth_error
                # print("Estimated depth = {}".format(estimated_distance))
                # print("Depth error = {}".format(depth_error))
                # print("average disparity = {}, calculated distance = {}".format(average_disparity, estimated_distance))

                # cv2.imshow("segmentation-object-"+str(i), segmentation_map_of_object)
                if math.isnan(estimated_distance):
                    print("NAN")
                    raise ValueError("Cannot have NAN!")
                else:
                    draw_bounding_box(
                        original_image, object_classes[class_ids[i]], estimated_distance, 
                        boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3],
                        (255, 178, 50)
                    )
            
        end_time = time.time()
        print("Timing overview".center(50, "-"))
        print("| Preprocessing time: {}".format(end_preprocessing_time - start_preprocessing_time))
        print("| Object detection time: {}".format(end_object_detection_time - start_object_detection_time))
        print("| Disparity calculation time: {}".format(end_disparity_calc_time - start_disparity_calc_time))
        if len(average_segmentation_time) > 0:
            print("| Average segmentation time: {}".format(sum(average_segmentation_time) / len(average_segmentation_time)))
        print("| Total time taken: {} ".format(end_time - start_preprocessing_time))

        # Output disparity
        cv2.namedWindow("calculated_disparity", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('calculated_disparity', 620, 348)
        cv2.imshow("calculated_disparity", calculated_disparity * 255)
        cv2.moveWindow("calculated_disparity", 0, 200)
        # cv2.imshow("calculated_disparity", calculated_disparity)
        # Output detected objects
        cv2.namedWindow("object_detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('object_detection', 1280, 720)
        cv2.moveWindow("object_detection", 630, 200)
        cv2.imshow("object_detection", original_image)

        cv2.namedWindow("original_image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('original_image', 620, 348)
        cv2.moveWindow("original_image", 0, 580)

        cv2.imshow("original_image", cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR))
        print("Total objects = {}".format(total_objects))
        cv2.waitKey(2000)

cv2.destroyAllWindows()