from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import mpl_toolkits
import pandas as pd
import unittest
import os
from parameterized import parameterized
import employee



class EmployeeModuleTest(unittest.TestCase):
    # =========================================================================
    # ==================== Checks the name of the employee ====================
    # =========================================================================

    @parameterized.expand(
        [
            ["nikhil","maram","maram_nikhil"],
            ["shruthi","menoth","menoth_shruthi"]
        ]
    )
    def test_user_name(self,first_name,last_name,expected):
        emp = employee.Employee()
        emp.set_first_name(first_name)
        emp.set_last_name(last_name)
        self.assertEqual( emp.get_name(),expected,"User names are same")


    # =========================================================================
    # ==================== Checks the employee id to employee dict ============
    # =========================================================================

    def test_employee_id_to_employee_from_file(self):
        employee_id_list = [1,2,3]
        employee_last_name_list = ["maram","maram","menoth"]
        employee_first_name_list = ["nikhil","jyothi","shruthi"]
        df = pd.DataFrame()
        df["el_id"] = employee_id_list
        df["last.name"] = employee_last_name_list
        df["first.name"] = employee_first_name_list
        file_path = "./tmp.xlsx"
        df.to_excel(file_path)
        employee_id_dict,_ = employee.employee_id_to_username_from_file(file_path)
        os.remove(file_path)
        expected_dict = {1:"maram_nikhil",2:"maram_jyothi",3:"menoth_shruthi"}
        self.assertDictEqual(employee_id_dict,expected_dict)

    # =========================================================================
    # ==================== Checks the employee to account mapping ============
    # =========================================================================

    def test_map_employee_account(self):
        employees_list = ["Wolfberg.Adam","Sedoy.Michael"]
        accounts_list = ["ADAM", "ALTM, MENG, ADAM"]
        df = pd.DataFrame()
        df["name"] = employees_list
        df["Account"] = accounts_list
        file_path = "./tmp.xlsx"
        df.to_excel(file_path)
        account_to_employee_dict, employee_to_account_dict = employee.map_employee_account(file_path)
        os.remove(file_path)
        account_to_employee_dict_expected = {"ADAM":["wolfberg_adam","sedoy_michael"],"ALTM" : ["sedoy_michael"],"MENG":["sedoy_michael"]}
        employee_to_account_dict_expected = {"wolfberg_adam":["ADAM"],"sedoy_michael":["ALTM","MENG","ADAM"]}
        self.assertDictEqual(account_to_employee_dict, account_to_employee_dict_expected)
        self.assertDictEqual(employee_to_account_dict,employee_to_account_dict_expected)

    # =========================================================================
    # ==================== Checks the address to user mapping =================
    # =========================================================================
    @parameterized.expand([
        ["rachacoso@diamondbackcap.com", "achacoso_ralph"],
        ["rachacosodbc", "achacoso_ralph"],
        ["aghania2@yahoo.com", "ameziane_ghania"],
        ["nikhilmaram@gmail.com", "nikhilmaram"]
    ])
    def test_lambda_func_user_address_mapping(self, address, user_expected):
        user = employee.lambda_func_user_address_mapping(address)
        self.assertEqual(user, user_expected, "Address Mapping Passed")

    # =========================================================================
    # ================Checks the user is inside/outside network =================
    # =========================================================================
    @parameterized.expand([
        ["ahmed_mustaque", 1],
        ["ameziane_ghania", 0],
        ["nikhilmaram", 0]

    ])
    def test_lambda_func_user_in_network(self, user_name, expected):
        observed = employee.lambda_func_user_in_network(user_name)
        self.assertEqual(observed, expected, "User inside/outside network Passed")

    # =========================================================================
    # ================Checks the employee hierarchy============================
    # =========================================================================

    def test_create_employee_hierarchy(self):
        e1 = employee.Employee() ; e1.set_first_name("n") ; e1.set_last_name("a") ; e1.set_supervisor("n_c")
        e2 = employee.Employee(); e2.set_first_name("n"); e2.set_last_name("b"); e2.set_supervisor("n_c")
        e3 = employee.Employee(); e3.set_first_name("n"); e3.set_last_name("c"); e3.set_supervisor("n_d")
        e4 = employee.Employee(); e4.set_first_name("n"); e4.set_last_name("d"); e4.set_supervisor("")
        e5 = employee.Employee(); e5.set_first_name("n"); e5.set_last_name("e"); e5.set_supervisor("")
        emp_dict = {"n_a":e1,"n_b":e2 ,"n_c":e3 , "n_d":e4, "n_e":e5}

        emp_dict = employee.create_employee_hierarchy(emp_dict)
        top_emp_obs = emp_dict["ROOT"]

        self.assertEqual(["n_d","n_e"],top_emp_obs.immediate_subordinates)
        self.assertEqual(["n_c"],emp_dict["n_d"].immediate_subordinates)
        self.assertEqual(["n_a","n_b"],emp_dict["n_c"].immediate_subordinates)
        self.assertEqual([], emp_dict["n_a"].immediate_subordinates)

    # =========================================================================
    # ================Checks the employee subordinates=========================
    # =========================================================================

    @parameterized.expand([
        ["ROOT",["ROOT","n_a","n_b","n_c","n_d","n_e"]],
        ["n_c",["n_a","n_b","n_c"]]
    ])
    def test_subordinates_given_employee(self,inp_emp_name,subordinate_list_exp):
        e1 = employee.Employee(); e1.set_first_name("n"); e1.set_last_name("a"); e1.set_supervisor("n_c")
        e2 = employee.Employee(); e2.set_first_name("n"); e2.set_last_name("b"); e2.set_supervisor("n_c")
        e3 = employee.Employee(); e3.set_first_name("n"); e3.set_last_name("c"); e3.set_supervisor("n_d")
        e4 = employee.Employee(); e4.set_first_name("n"); e4.set_last_name("d"); e4.set_supervisor("")
        e5 = employee.Employee(); e5.set_first_name("n"); e5.set_last_name("e"); e5.set_supervisor("")
        emp_dict = {"n_a": e1, "n_b": e2, "n_c": e3, "n_d": e4, "n_e": e5}

        emp_dict = employee.create_employee_hierarchy(emp_dict)
        subordinates_list_obs = employee.subordinates_given_employee(emp_dict,inp_emp_name)

        self.assertEqual(subordinate_list_exp,subordinates_list_obs)

    # =========================================================================
    # ================Checks the book list generated===========================
    # =========================================================================

    def test_books_given_employee_list(self):
        book_list_obs = employee.books_given_employee_list(["wolfberg_adam","fishman_mark","anfang_mark"])
        book_list_exp = ["ADAM","DBIG","FISH"]
        self.assertEqual(book_list_exp,book_list_obs)


if __name__ == '__main__':
    unittest.main()