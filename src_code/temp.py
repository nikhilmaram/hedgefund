
"""Only to run files."""

import employee
import config as cfg
import processing_all_files

address_to_user_dict,user_to_address_dict =  employee.map_user_address(cfg.ADDRESS_LINK_FILE)
employee_dict = employee.get_emplpoyees_from_file(cfg.EMPLOYEE_MASTER_FILE)

print("Employees in Address File : {0}".format(len(user_to_address_dict.keys())))
print("Employees in Master File  : {0}".format(len(employee_dict.keys())))

print(user_to_address_dict.keys())
print(employee_dict.keys())

