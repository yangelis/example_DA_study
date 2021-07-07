import xline as xl
import xtrack as xt

line = xl.Line.from_json('../000_machine_model/xline/line_bb_dipole_not_cancelled.json')




tracker = xt.Tracker(sequence=line)



  
