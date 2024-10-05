from pydantic import BaseModel, Field
from typing import Optional, List

class navigation(BaseModel):
    """Navigate to the target location"""

    target_name: str = Field(..., description="the name of the target location")

class manipulation(BaseModel):
    """Manipulate the target object"""

    target_name: str = Field(..., description="the name of the target object")

class structure_json_output(BaseModel):
    """format the question to json format base on the arg provided"""




#region ######## JSON Structure Output Format #########

## Object List
class Object(BaseModel):
    '''detail break down of item'''

    name: str = Field(..., description="the name of the object detected")
    position: List[int] = Field(..., description="Return bounding boxes as JSON arrays [ymin, xmin, ymax, xmax]")
    features: List[str] = Field(..., description="the key features of the object detected")
    # id: int = Field(..., description="id of this object")

class ObjectList(BaseModel):
    '''a list of items description'''

    objects: List[Object] = Field(..., description="the list of items")
    description: str = Field(..., description="overall description of what is seen in the image")

## Array of item
class ListItems(BaseModel):
    '''list of items'''

    items: List[str] = Field(..., description="the list of items")

## 

#endregion