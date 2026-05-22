// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AlertLogger {
    struct Alert {
        string timestamp;
        string alertType;
        string sourceIP;
        string destIP;
    }

    Alert[] public alerts;

    event AlertLogged(string timestamp, string alertType, string sourceIP, string destIP);

    function logAlert(string memory timestamp, string memory alertType, string memory sourceIP, string memory destIP) public {
        alerts.push(Alert(timestamp, alertType, sourceIP, destIP));
        emit AlertLogged(timestamp, alertType, sourceIP, destIP);
    }

    function getAlertCount() public view returns (uint) {
        return alerts.length;
    }
}
