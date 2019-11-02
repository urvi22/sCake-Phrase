#include <bits/stdc++.h>
using namespace std;

int main()
{
	int t;
	cin >> t;

	while (t--)
	{
		string n;
		cin >> n;

		int l = n.length();

		int i = 0;

		for (i=1; i<n; i++)
		{
			if (n[i]<n[i-1])
			{
				break;
			}
		}

		if (i==n)
		{
			i--;
		}

		bool f=0;

		for (int j=0; j<n; j++)
		{
			if (!f && j>=i)
			{
				f=1;
				j++;

				while (j<n && n[j]=='0')
				{
					j++;
				}
				continue;
			}
			cout << n[j];
		}

		cout << '\n';
	}

	return 0;
}
