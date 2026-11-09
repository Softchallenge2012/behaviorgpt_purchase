<table>
  <tr>
    <td align="center">
      <img src="examples/ui_v0.png" width="100%" alt="Image 1 Description">
      <br>
      <sub><b>Figure 1:</b> Load Page</sub>
    </td>
    <td align="center">
      <img src="examples/ui_v1.png" width="100%" alt="Image 2 Description">
      <br>
      <sub><b>Figure 2:</b> Load Results</sub>
    </td>
  </tr>
</table>

We use Amazon grocery review data to model and predict future shopping behavior. First, a user's purchase history is selected, and a catalog is uploaded to [portal.unboxai.com](https://portal.unboxai.com). 

Given a sequence of purchases within a specific time window, the system predicts what the user will buy next. Finally, we benchmark these predictions against the customer's actual purchases within the time window.

**How to run**

cd examples

python app.py
