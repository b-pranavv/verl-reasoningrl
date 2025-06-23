import json
import re
from verl.workers.rollout.vllm_rollout.python_executor_sandbox import executeCode
from verl.workers.rollout.vllm_rollout.bfcl_executor import BfclExecutor


class ToolRouter:
    def __init__(self, complete_solution_str, output_str):

        self.complete_solution_str = complete_solution_str ## complete solution string is required for stateful tool calls
        self.output_str = output_str  ## output string is required for stateless tool calls
        
        self.tool_calls = self.extract_tool_calls(complete_solution_str)
        self.current_tool_call = self.extract_tool_calls(output_str)
    
    def validate_tool_calls(self, output_str):
        start_tags = re.findall(r'<tool_call>', output_str)
        end_tags = re.findall(r'</tool_call>', output_str)
        
        if len(start_tags) != len(end_tags):
            return False
            
        start_positions = [m.start() for m in re.finditer(r'<tool_call>', output_str)]
        end_positions = [m.start() for m in re.finditer(r'</tool_call>', output_str)]
        
        for start, end in zip(start_positions, end_positions):
            if start >= end:
                return False
                
        return True

    def extract_tool_calls(self, output_str):
        if not self.validate_tool_calls(output_str):
            return []

        try:
            pattern = r'<tool_call>((?:(?!</tool_call>).)*)</tool_call>'
            matches = re.finditer(pattern, output_str, re.DOTALL)
            
            return [match.group(1).strip() for match in matches]
        except Exception as e:
            return []
        
    
    def tool_call_present(self):
        """
        Check if the solution string contains tool calls in the format <tool_call> </tool_call>
        """
        return len(self.current_tool_call) > 0
    
    def is_json(self, s):
        # print(f"string in is_json: {s}[string end]")
        try:
            json.loads(s)
            return True
        except Exception as e:
            # print(e)
            # print(f'not a json string: {s}[string end]')
            try:
                s = json.dumps(eval(s))
                json.loads(s)
                return True
            except:
                return False
            
    def parse_json(self, s):
        try:
            data = json.loads(s)
        except ValueError:
            s = json.dumps(eval(s))
            data = json.loads(s)

        return data
    
    def parse_tools(self, tool_call):
        if self.is_json(tool_call):
            res = self.parse_json(tool_call)
            if 'name' not in res or 'arguments' not in res:
                print(f"Wrong tool call result: {res}")
                return {'type': 'error', 'message': f'Wrong tool call result: {res}'}
            tool_name = res['name']
            if isinstance(res['arguments'], dict):
                arguments = res['arguments']
                tool_category = self.get_tool_category(tool_name)
                if tool_category is None:
                    return {'type': 'error', 'message': f'Invalid tool: {tool_name}'}
                return {'type': 'tool', 'tool_category': tool_category, 'tool_name': tool_name, 'arguments': arguments}
            elif self.is_json(res['arguments']):
                    arguments = self.parse_json(res['arguments'])
                    tool_category = self.get_tool_category(tool_name)
                    if tool_category is None:
                        return {'type': 'error', 'message': f'Invalid tool: {tool_name}'}
                    return {'type': 'tool', 'tool_category': tool_category, 'tool_name': tool_name, 'arguments': arguments}
            else:
                print(f"Wrong argument format: {res['arguments']}")
                return {'type': 'error', 'message': f"Wrong argument format: {res['arguments']}"}
        else:
            print(f"Wrong tool call completion result: {tool_call}")
            return {'type': 'error', 'message': f'Wrong tool call format: {tool_call}'}
        
        

    def get_tool_category(self, tool_name):
        """
        Extract the category of the tool call based on its name.
        """
        if tool_name == 'run_python':
            return 'python'
        elif tool_name in ['cat', 'cd', 'cp', 'diff', 'du', 'echo', 'find', 'grep', 'ls', 'mkdir', 'mv', 'pwd', 'rm', 'rmdir', 'sort', 'tail', 'touch', 'wc', 'absolute_value', 'add', 'divide', 'imperial_si_conversion', 'logarithm', 'max_value', 'mean', 'min_value', 'multiply', 'percentage', 'power', 'round_number', 'si_unit_conversion', 'square_root', 'standard_deviation', 'subtract', 'sum_values', 'add_contact', 'delete_message', 'get_message_stats', 'get_user_id', 'list_users', 'message_get_login_status', 'message_login', 'search_messages', 'send_message', 'view_messages_sent', 'authenticate_twitter', 'comment', 'follow_user', 'get_tweet', 'get_tweet_comments', 'get_user_stats', 'get_user_tweets', 'list_all_following', 'mention', 'post_tweet', 'posting_get_login_status', 'retweet', 'search_tweets', 'unfollow_user', 'close_ticket', 'create_ticket', 'edit_ticket', 'get_ticket', 'get_user_tickets', 'logout', 'resolve_ticket', 'ticket_get_login_status', 'ticket_login', 'add_to_watchlist', 'cancel_order', 'filter_stocks_by_price', 'fund_account', 'get_account_info', 'get_available_stocks', 'get_current_time', 'get_order_details', 'get_order_history', 'get_stock_info', 'get_symbol_by_name', 'get_transaction_history', 'get_watchlist', 'make_transaction', 'notify_price_change', 'place_order', 'remove_stock_from_watchlist', 'trading_get_login_status', 'trading_login', 'trading_logout', 'update_market_status', 'update_stock_price', 'authenticate_travel', 'book_flight', 'cancel_booking', 'compute_exchange_rate', 'contact_customer_support', 'get_all_credit_cards', 'get_budget_fiscal_year', 'get_credit_card_balance', 'get_flight_cost', 'get_nearest_airport_by_city', 'list_all_airports', 'purchase_insurance', 'register_credit_card', 'retrieve_invoice', 'set_budget_limit', 'travel_get_login_status', 'verify_traveler_information', 'activateParkingBrake', 'adjustClimateControl', 'check_tire_pressure', 'displayCarStatus', 'display_log', 'estimate_distance', 'estimate_drive_feasibility_by_mileage', 'fillFuelTank', 'find_nearest_tire_shop', 'gallon_to_liter', 'get_current_speed', 'get_outside_temperature_from_google', 'get_outside_temperature_from_weather_com', 'get_zipcode_based_on_city', 'liter_to_gallon', 'lockDoors', 'pressBrakePedal', 'releaseBrakePedal', 'setCruiseControl', 'setHeadlights', 'set_navigation', 'startEngine']:
            return 'bfcl'
        else:
            return None
    
        
    def execute_tool_calls(self, env=None):
        parsed_output = self.parse_tools(self.current_tool_call)
        
        if (parsed_output['type'] == 'error'):
            return {'success': False, 'message': parsed_output['message']}
        
        elif parsed_output['type'] == 'tool':
            tool_category = parsed_output['tool_category']
            
            if tool_category == 'python':
                return self.execute_python_tool(parsed_output['arguments'])
            elif tool_category == 'bfcl':
                parsed_output_list = [self.parse_tools(tool_call) for tool_call in self.tool_calls]
                tool_call_list = [tool_call for tool_call in parsed_output_list if tool_call['type'] == 'tool' and tool_call['tool_category'] == 'bfcl']
                class_list = self.extract_classes(self.complete_solution_str)
                
                return self.execute_bfcl_tool(tool_call_list, class_list, env)
            else:
                return {'success': False, 'message': f'Unknown tool category: {tool_category}'}
        else:
            return {'success': False, 'message': f'Invalid tool call type: {parsed_output["type"]}'}
    
    
    def execute_python_tool(self, code, env=None):
        
        if 'code' not in code:
            return {'success': False, 'message': 'Invalid code format. Expected a dictionary with a "code" key.'}
        code = code['code']
        
        code_execution = executeCode([code])
        
        code_execution = code_execution[0]
        
        success = code_execution['success']
        
        if(success):
            return {'success': True, 'output': code_execution['output']}
        else:
            return {'success': False, 'message': code_execution['error']}
    
    def execute_bfcl_tool(self, tool_call_list, class_list, env):
        bfcl_execution = BfclExecutor(timeout_seconds=8, capture_stdout=True)
        
        # convert tool_call_list to list of jsonl
        tool_calls = [json.dumps(tool_call) for tool_call in tool_call_list]
        tool_calls = str(tool_calls).replace("'", '"')

        response, _ = bfcl_execution.execute_list(tool_calls, class_list, env)

        return response[-1]
    
    def extract_classes(self, input_str):
        """Extract class names from the input string, it would be present as [Classes Involved: {classes_involved}]"""
        pattern = r'\[Classes Involved: (.*?)\]'
        match = re.search(pattern, input_str)
        if match:
            classes_str = match.group(1)
            classes_str_alphabets = re.sub(r'[^a-zA-Z\s]', '', classes_str)
            classes_list = [cls.strip() for cls in classes_str_alphabets.split(' ')]
            return classes_list
        else:
            return []