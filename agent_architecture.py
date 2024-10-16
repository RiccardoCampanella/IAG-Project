

from owlready2 import * 
import owlready2
owlready2.JAVA_EXE = r"C:\Users\lucmi\Downloads\Protege-5.6.4-win\Protege-5.6.4\jre\bin\java.exe"
# Create a Graph \Documents\Opdrachten\IAG-Project
onto = get_ontology(r"C:\Users\lucmi\Documents\Opdrachten\IAG-Project\intelligent_agents.rdf").load()
#print(type(onto))
#for entity in onto.entities():
#    print(entity)
with onto:
    sync_reasoner()
#for cls in onto.classes():
 #   print("yeah", cls)
ar1 = onto.Healthy.instances()
ar2 = onto.Sport.instances()
inte = list(set(ar1).intersection(ar2))
print("result", inte)
#for health in onto.Healthy.instances():
#    for sport in onto.Sport.instances():
 #       print("healthy thing", health, "sport", sport)

