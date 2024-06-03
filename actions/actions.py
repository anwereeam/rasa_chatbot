from typing import Any, Text, Dict, List
import joblib
import pandas as pd
import random
from rasa_sdk.events import SlotSet
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher,Tracker
from rasa_sdk import ActionExecutionRejection

#
#
####################---select symptom action ------########################        
class ActionSelect(Action):
    def name(self) -> Text:
        return "action_select_symptom"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker,domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        symptoms = tracker.get_slot("symptoms")
        selected_symptom = tracker.get_slot("selected")
        filter = tracker.get_slot("filter")
        i = 0
        l=0
        sympa=[]
        if filter:
            filters = pd.read_csv('F://final_project//new_dataset//rasa1//actions//data.csv')
            while sympa==[]:
                sympa = filters[filters['Disease'] == filter[i]].drop('Disease', axis=1)
                sympa = sympa.columns[sympa.iloc[0].values == 1].tolist()
            while sympa[l] not in symptoms:
                del sympa[l]
            if sympa[l] == sympa[-1]:
                del filter[i]
            sym=sympa[l]
            selected_symptom.append(sym)
            symptoms.remove(sym)
        else:
            if not symptoms:
                dispatcher.utter_message("Sorry, I didn't understand your disease")
                return []
            else:
                sym = random.choice(symptoms)
                selected_symptom.append(sym)
                symptoms.remove(sym)
        return [SlotSet("selected", selected_symptom), SlotSet("symptoms", symptoms),SlotSet("filter", filter)]
####################---ask user action ------########################        
class ActionAskDisease(Action):
    def name(self) -> Text:
        return "action_ask_disease"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker,domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        selected_symptom = tracker.get_slot("selected")
        dispatcher.utter_message("Do you suffer from {symp} ?".format(symp=selected_symptom[-1]))
        return []

####################---predict disease action ------########################        
class ActionPredictDisease(Action):
    def name(self) -> Text:
        return "action_predict_disease"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker,domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        model = joblib.load('F://final_project//new_dataset//rasa1//actions//mlp_model')
        symptom_frame = pd.read_csv('F://final_project//new_dataset//rasa1//actions//patient.csv')
        symptoms_model=tracker.get_slot("symptoms_model")
        symptom_frame[symptoms_model]=1
        predicted_proba = model.predict_proba(symptom_frame)[0]
        possible_diseases = model.classes_
        predictions = {}
        for disease, proba in zip(possible_diseases, predicted_proba):
            predictions[disease] = proba
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        first_value = sorted_predictions[0]
        if first_value[1] > .9:
            dispatcher.utter_message("it seems like you might be suffering from {dis}. Please consult a healthcare professional for further evaluation.".format(dis=first_value[0]))
            return [SlotSet("condition", 1),SlotSet("dis", first_value[0])]
        else:
            filter = tracker.get_slot("filter")
            filter.append(sorted_predictions[0][0])
            filter.append(sorted_predictions[1][0])
            filter.append(sorted_predictions[2][0])
            #dispatcher.utter_custom_json({"trigger_custom_action": "action_select_symptom"})
            return [SlotSet("condition", 0), SlotSet("filter", filter)]
####################---yes action ------########################        
class ActionYes(Action):
    def name(self) -> Text:
        return "action_yes"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker,domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        selected_symptom = tracker.get_slot("selected")
        symptoms_model = tracker.get_slot("symptoms_model")
        symptoms_model.append(selected_symptom[-1])
        #dispatcher.utter_custom_json({"trigger_custom_action": "action_predict_disease"})
        return [SlotSet("symptoms_model", symptoms_model)]
    
####################---provide information action ------########################
class ActionProvideInfo(Action):
    def name(self) -> Text:
        return "action_provide_info"
    def run(self, dispatcher: CollectingDispatcher,tracker: Tracker,domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dis = tracker.get_slot("dis")
        info = pd.read_csv('F://final_project//new_dataset//rasa1//actions//info1.csv')
        symp = pd.read_csv('F://final_project//new_dataset//rasa1//actions//patient.csv')
        disease = info[info['disease'].str.contains(dis, case=False)]
        tip1 = disease['tips'].iloc[0] if not disease['tips'].empty else ""
        tip2 = disease['tips2'].iloc[0] if not disease['tips2'].empty else ""
        dispatcher.utter_message(f"Here are some information for {dis}: {tip1}")
        dispatcher.utter_message(f"{tip2}")  
        return [SlotSet("symptoms",symp.columns.tolist()),SlotSet("symptoms_model",[])]

        

        