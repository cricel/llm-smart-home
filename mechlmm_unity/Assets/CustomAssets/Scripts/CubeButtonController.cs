using UnityEngine;

public class CubeButtonController : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void OnTriggerEnter(Collider other)
    {
        Debug.Log("--");
        if(other.name == "Trigger Point")
        {
            Debug.Log(other.name);
        }
    }
}